import os
import sys
import time

import numpy as np
import pandas as pd
import cv2
import torch
import func_timeout

from reference_image_mapper.models.two_view_pipeline import TwoViewPipeline
from reference_image_mapper.tools import numpy_image_to_torch, batch_to_np
import reference_image_mapper

 
def mapCoords2D(coords, transform2D):
    coords = np.array(coords, dtype=np.float32).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(coords, transform2D)
    mapped = np.round(mapped.ravel(), 3)
    return float(mapped[0]), float(mapped[1])


def _ensure_int64(df, col):
    df[col] = pd.to_numeric(df[col], errors="raise", downcast=None).astype("int64")
    return df
 
    
def preprocessing_rim(gaze_data, time_stamps, config):
    """
    Retourne un DF gaze avec:
      - timestamp [ns] : int64
      - frame_idx      : int64 (index frame dans la vidéo)
      - norm_pos_x/y   : float64
      - confidence     : float32

    Si time_stamps est None / non fourni, frame_idx est estimé via fps_scene.
    """
    import numpy as np
    import pandas as pd

    # IMPORTANT: forcer dtype int64 à la lecture => pas de float => pas de perte de précision
    gaze_df = pd.read_csv(gaze_data, dtype={'timestamp [ns]': 'int64'})

    # sécurité + tri
    gaze_df = gaze_df.sort_values('timestamp [ns]').reset_index(drop=True)

    per_n = int(config['processing'].get('downsampling_factor', 1))
 
    if time_stamps is not None and str(time_stamps).strip() != "":
        world_df = pd.read_csv(time_stamps, dtype={'timestamp [ns]': 'int64'})
        world_df = world_df.sort_values('timestamp [ns]').reset_index(drop=True)

        world_df = world_df.copy()
        world_df['frame_idx'] = np.arange(len(world_df), dtype=np.int64)
        world_ds = world_df.iloc[::per_n].copy()

        tolerance_ns = config['processing'].get('timestamp_tolerance_ns', None)
        merged = pd.merge_asof(
            gaze_df,
            world_ds[['timestamp [ns]', 'frame_idx']],
            on='timestamp [ns]',
            direction='nearest',
            tolerance=tolerance_ns
        )

        # si tolérance activée: compléter les NaN au plus proche sans tolérance
        if merged['frame_idx'].isna().any():
            nan_mask = merged['frame_idx'].isna()
            merged_no_tol = pd.merge_asof(
                gaze_df.loc[nan_mask, ['timestamp [ns]']].copy(),
                world_ds[['timestamp [ns]', 'frame_idx']],
                on='timestamp [ns]',
                direction='nearest',
                tolerance=None
            )
            merged.loc[nan_mask, 'frame_idx'] = merged_no_tol['frame_idx'].to_numpy()

        frame_idx_arr = merged['frame_idx'].astype('int64').to_numpy()
 
    else:
        fps_scene = float(config['processing'].get('fps_scene', 30.0))
    
        gaze_ts = gaze_df['timestamp [ns]'].to_numpy(dtype=np.int64)
    
        fps_num = int(round(fps_scene * 1000))
        fps_den = 1000
    
        t0_ns = config['processing'].get('t0_ns', None)
        if t0_ns is None:
            t0_ns = int(gaze_ts[0])
        else:
            t0_ns = int(t0_ns)
    
        dt = gaze_ts - np.int64(t0_ns)
        dt = np.clip(dt, 0, None).astype(np.int64)
    
        denom = np.int64(1_000_000_000) * np.int64(fps_den)
    
        frame_idx_est = (dt * np.int64(fps_num)) // denom
        frame_idx_est = frame_idx_est.astype(np.int64)
    
        per_n = int(config['processing'].get('downsampling_factor', 1))
     
        frame_idx_arr = (frame_idx_est // np.int64(per_n)) * np.int64(per_n)
        frame_idx_arr = frame_idx_arr.astype(np.int64)


    # normalisation gaze
    cam_w = float(config['processing']['camera']['width'])
    cam_h = float(config['processing']['camera']['height'])
    gaze_x = gaze_df['gaze x [px]'].astype(np.float64) / cam_w
    gaze_y = gaze_df['gaze y [px]'].astype(np.float64) / cam_h
    confidence = gaze_df['worn'].astype(np.float32)

    out = pd.DataFrame({
        'timestamp [ns]': gaze_df['timestamp [ns]'].astype('int64'),
        'frame_idx': frame_idx_arr.astype('int64'),
        'confidence': confidence,
        'norm_pos_x': gaze_x,
        'norm_pos_y': gaze_y
    })

    # sanity checks
    assert out['timestamp [ns]'].dtype == np.int64
    assert out['frame_idx'].dtype == np.int64

    return out



# -------------------------
# 2) Processing vidéo + mapping world->ref
# -------------------------
 
def processRecording(gazeWorld_df,
                     reference_image,
                     world_camera,
                     out_name,
                     config,
                     gaze_data_path=None,
                     warm_start=0):

    start_time = time.time()
    outputDir = config['processing']['files']['outputDir']
    os.makedirs(outputDir, exist_ok=True)

    framesToCompute = gazeWorld_df['frame_idx'].astype(np.int64).tolist()
    last_frame = int(max(framesToCompute))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipeline_model = TwoViewPipeline(config['model']).to(device).eval()

    d_w = int(config['processing']['camera']['down_width'])
    d_h = int(config['processing']['camera']['down_height'])
    down_points = (d_w, d_h)

    ref_orig = cv2.imread(reference_image, cv2.IMREAD_GRAYSCALE)
    if ref_orig is None:
        raise FileNotFoundError(f"Reference image not found: {reference_image}")

    ref_h_orig, ref_w_orig = ref_orig.shape[:2]
    ref_frame = cv2.resize(ref_orig, down_points, interpolation=cv2.INTER_LINEAR)
    torch_ref = numpy_image_to_torch(ref_frame)

    with torch.no_grad():
        pred = pipeline_model({
            'image0': torch_ref[None].to(device),
            'image1': torch_ref[None].to(device)
        })

    pred_ref = {k[:-1]: pred[k] for k in [
        'keypoints1', 'keypoint_scores1', 'descriptors1',
        'pl_associativity1', 'num_junctions1', 'lines1',
        'orig_lines1', 'lines_junc_idx1', 'line_scores1',
        'valid_lines1'
    ]}

    gazeMapped_rows = []

    vid = cv2.VideoCapture(world_camera)
    if not vid.isOpened():
        raise FileNotFoundError(f"World camera video not found/cannot open: {world_camera}")

    world2ref_transform = None
    frameCounter = 0
    m_time = config['processing']['max_time']

    print("Processing frames...")

    framesToCompute_set = set(framesToCompute)

    while vid.isOpened():
        ret, frame = vid.read()
        if not ret or frameCounter > last_frame:
            break

        if frameCounter in framesToCompute_set and frameCounter >= warm_start:
            world_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            world_gray = cv2.resize(world_gray, down_points)

            valid = True
            n_matches = 0

            try:
                torch_world = numpy_image_to_torch(world_gray)
                pred = func_timeout.func_timeout(
                    m_time,
                    pipeline_model,
                    args=[{
                        'image0': torch_world[None].to(device),
                        'image1': torch_ref[None].to(device),
                        'ref': pred_ref
                    }]
                )
                pred = batch_to_np(pred)
            except Exception:
                valid = False

            if valid:
                try:
                    m0 = pred['matches0']
                    valid_idx = m0 != -1
                    n_matches = int(valid_idx.sum())

                    kp0 = pred['keypoints0'][valid_idx]
                    kp1 = pred['keypoints1'][m0[valid_idx]]

                    H, _ = cv2.findHomography(kp1, kp0, cv2.RANSAC, 10)
                    if H is None:
                        valid = False
                        world2ref_transform = None
                    else:
                        world2ref_transform = cv2.invert(H)[1]
                except Exception:
                    valid = False
                    world2ref_transform = None

            sys.stdout.write(f"\rFrame {frameCounter}/{last_frame} | matches={n_matches}")
            sys.stdout.flush()

            # IMPORTANT: on écrit TOUJOURS les lignes gaze de cette frame
            rows = gazeWorld_df[gazeWorld_df['frame_idx'] == frameCounter]

            for r in rows.itertuples(index=False):
                ts = np.int64(getattr(r, 'timestamp [ns]') if hasattr(r, 'timestamp [ns]') else r[0])

                world_gx = float(r.norm_pos_x * config['processing']['camera']['width'])
                world_gy = float(r.norm_pos_y * config['processing']['camera']['height'])

                if world2ref_transform is None:
                    # Pas de transform => ref à 0, matches à 0, mapped False
                    gazeMapped_rows.append((
                        np.int64(ts),
                        np.int64(frameCounter),
                        world_gx,
                        world_gy,
                        0.0,          # ref_gazeX
                        0.0,          # ref_gazeY
                        np.int64(0),  # number_point_matches
                        False         # mapped
                    ))
                else:
                    # Transform dispo => mapping normal (mapped = valid)
                    model_x = float(r.norm_pos_x) * d_w
                    model_y = float(r.norm_pos_y) * d_h
                    ref_x_model, ref_y_model = mapCoords2D((model_x, model_y), world2ref_transform)

                    ref_x = ref_x_model * (ref_w_orig / d_w)
                    ref_y = ref_y_model * (ref_h_orig / d_h)

                    gazeMapped_rows.append((
                        np.int64(ts),
                        np.int64(frameCounter),
                        world_gx,
                        world_gy,
                        float(ref_x),
                        float(ref_y),
                        np.int64(n_matches),
                        bool(valid)
                    ))

        frameCounter += 1

    vid.release()

    gazeMapped_df = pd.DataFrame(
        gazeMapped_rows,
        columns=[
            'timestamp [ns]', 'worldFrame',
            'world_gazeX', 'world_gazeY',
            'ref_gazeX', 'ref_gazeY',
            'number_point_matches', 'mapped'
        ]
    )
 
    gazeMapped_df['timestamp [ns]'] = gazeMapped_df['timestamp [ns]'].astype('int64')
    gazeMapped_df['worldFrame'] = gazeMapped_df['worldFrame'].astype('int64')
    gazeMapped_df['number_point_matches'] = gazeMapped_df['number_point_matches'].astype('int64')
    gazeMapped_df['mapped'] = gazeMapped_df['mapped'].astype(bool)

    if gaze_data_path is not None:
        gaze_raw = pd.read_csv(
            gaze_data_path,
            dtype={'timestamp [ns]': 'int64'}
        )
     
        full_df = pd.DataFrame({
            'timestamp [ns]': gaze_raw['timestamp [ns]'].astype('int64'),
            'world_gazeX': gaze_raw['gaze x [px]'].astype(np.float64),
            'world_gazeY': gaze_raw['gaze y [px]'].astype(np.float64),
        })
     
        merged = full_df.merge(
            gazeMapped_df[[
                'timestamp [ns]', 'worldFrame',
                'ref_gazeX', 'ref_gazeY',
                'number_point_matches', 'mapped'
            ]],
            on='timestamp [ns]',
            how='left'
        )
     
        merged['ref_gazeX'] = merged['ref_gazeX'].fillna(0.0)
        merged['ref_gazeY'] = merged['ref_gazeY'].fillna(0.0)
        merged['number_point_matches'] = merged['number_point_matches'].fillna(0).astype('int64')
     
        merged['mapped'] = merged['mapped'].fillna(False).astype(bool)
     
        if gazeMapped_df.shape[0] >= 2: 
            base = gazeMapped_df[['timestamp [ns]', 'worldFrame']].dropna().copy()
            base = base.sort_values('timestamp [ns]') 
            base = base.drop_duplicates('timestamp [ns]', keep='last')
    
            x = base['timestamp [ns]'].to_numpy(dtype=np.int64)
            y = base['worldFrame'].to_numpy(dtype=np.int64)
     
            xi = merged['timestamp [ns]'].to_numpy(dtype=np.int64).astype(np.float64)
            x_f = x.astype(np.float64)
            y_f = y.astype(np.float64)
    
            interp_frames = np.interp(xi, x_f, y_f)     # float
            interp_frames = np.rint(interp_frames)      # arrondi
            interp_frames = interp_frames.astype(np.int64)
     
            wf = merged['worldFrame']
            missing_wf = wf.isna()
            merged.loc[missing_wf, 'worldFrame'] = interp_frames[missing_wf.to_numpy()]
    
        else:
       
            merged['worldFrame'] = merged['worldFrame'].fillna(0)
    
        merged['worldFrame'] = merged['worldFrame'].astype('int64')
     
        gazeMapped_full_df = merged[[
            'timestamp [ns]',
            'worldFrame',
            'world_gazeX', 'world_gazeY',
            'ref_gazeX', 'ref_gazeY',
            'number_point_matches',
            'mapped'
        ]].copy()
     
        out_full_csv = f"{outputDir}/mappedGaze_{out_name}.csv"
        gazeMapped_full_df.to_csv(out_full_csv, index=False)
        print(f"CSV saved to: {out_full_csv}")



def display_results(world_camera,
                    reference_image,
                    out_name,
                    out_dir,
                    down_width, down_height):

    csv_path = f"{out_dir}/mappedGaze_{out_name}.csv"
    out_video_path = f"{out_dir}/video_rim.avi"

    print("Saving video to:", os.path.abspath(out_video_path))

    # IMPORTANT: relire en forçant int64
    df = pd.read_csv(
        csv_path,
        dtype={'timestamp [ns]': 'int64', 'worldFrame': 'int64', 'number_point_matches': 'int64'}
    )

    cap = cv2.VideoCapture(world_camera)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sx = down_width / max(orig_w, 1)
    sy = down_height / max(orig_h, 1)

    ref_orig = cv2.imread(reference_image)
    if ref_orig is None:
        cap.release()
        raise FileNotFoundError(f"Reference image not found: {reference_image}")

    ref_h, ref_w = ref_orig.shape[:2]
    ref_resized = cv2.resize(ref_orig, (down_width, down_height))

    writer = cv2.VideoWriter(
        out_video_path,
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (down_width * 2, down_height)
    )
    if not writer.isOpened():
        print("ERROR: VideoWriter could not be opened")
        cap.release()
        return

    frame_idx = 0
    written = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (down_width, down_height))
        combined = np.hstack([frame_resized, ref_resized.copy()])

        rows = df[df['worldFrame'] == frame_idx]
        if not rows.empty:
            wx = np.median(rows['world_gazeX'].to_numpy()) * sx
            wy = np.median(rows['world_gazeY'].to_numpy()) * sy
            rx = np.median(rows['ref_gazeX'].to_numpy()) * (down_width / ref_w)
            ry = np.median(rows['ref_gazeY'].to_numpy()) * (down_height / ref_h)

            if not np.isnan([wx, wy, rx, ry]).any():
                cv2.circle(combined, (int(wx), int(wy)), 5, (0, 0, 255), -1)
                cv2.circle(combined, (int(rx) + down_width, int(ry)), 5, (255, 0, 0), -1)

        writer.write(combined)
        written += 1
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Video written successfully: {written} frames")


# -------------------------
# 4) Wrapper principal
# -------------------------
def process_rim(gaze_data,
                world_camera, reference_image,
                camera_width, camera_height,
                world_timestamps = None,
                **kwargs):

    out_name = kwargs.get('outfile_name', 'gaze')
    output_dir = kwargs.get('outdir_name', 'mappedGazeOutput')
    os.makedirs(output_dir, exist_ok=True)

    downsampling_factor = int(kwargs.get('downsampling_factor', 2))
    down_width = int(kwargs.get('down_width', 600))
    down_height = int(kwargs.get('down_height', 450))
    fps_scene = int(kwargs.get('fps_scene', 10))

    display_rim_results = bool(kwargs.get('display_rim_results', True))

    # Optionnel: tolérance de matching timestamps (ns)
    # Ex: à 30 fps, ~33ms => 16ms ~ 16_000_000 ns
    timestamp_tolerance_ns = kwargs.get('timestamp_tolerance_ns', None)

    config = {
        'processing': {
            'downsampling_factor': downsampling_factor,
            'max_time': 5,
            'fps_scene': fps_scene,
            'timestamp_tolerance_ns': timestamp_tolerance_ns,
            'camera': {
                'width': int(camera_width),
                'height': int(camera_height),
                'down_width': down_width,
                'down_height': down_height
            },
            'files': {
                'outputDir': output_dir
            }
        },
        'model': {
            'name': 'two_view_pipeline',
            'use_lines': True,
            'use_lines_homoraphy': False,
            'extractor': {
                'name': 'wireframe',
                'sp_params': {
                    'force_num_keypoints': False,
                    'max_num_keypoints': 3000,
                },
                'wireframe_params': {
                    'merge_points': True,
                    'merge_line_endpoints': True,
                },
                'max_n_lines': 0,
            },
            'matcher': {
                'name': 'gluestick',
                'weights': str(reference_image_mapper.GLUESTICK_ROOT /
                               'resources/weights/checkpoint_GlueStick_MD.tar'),
                'trainable': False,
            },
            'ground_truth': {
                'from_pose_depth': False,
            }
        }
    }

    preProData = preprocessing_rim(gaze_data, world_timestamps, config)
 
    processRecording(preProData,
                     reference_image,
                     world_camera,
                     out_name,
                     config,
                     gaze_data_path=gaze_data
                     )

    if display_rim_results:
        display_results(world_camera, 
                        reference_image, 
                        out_name, output_dir, 
                        down_width, down_height)
        
        
        
        
        