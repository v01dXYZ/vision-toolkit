# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from string import ascii_uppercase as ABC

import pandas as pd
import seaborn as sns
import plotly.figure_factory as ff
 
from plotly.subplots import make_subplots


import vision_toolkit as v


def plot_scanpaths_with_grid_arrows( 
    xlim=(0, 1200), ylim=(0, 800),
    x_splits=None, y_splits=None,
    save_path='figures/scanpath_grid_2.png',
    arrow_lw=1.6, arrow_scale=18, arrowstyle="-|>"
):
    if x_splits is None:
        x_splits = np.linspace(xlim[0], xlim[1], 5)  # 4 colonnes
    if y_splits is None:
        y_splits = np.linspace(ylim[0], ylim[1], 5)  # 4 lignes

    blue_xy = np.array([
        [888,701],[610,501],[500,550],[527,701],
        [1020,500],[447,267],[200,51],[217,317],[401,101]
    ], dtype=float)

    purple_xy = np.array([
        [995,760],[405,505],[560,640],[820,660],
        [1060,670],[915,425],[680,415],[342,220],[90,220],
        [265,105],[405,20]
    ], dtype=float)

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots()

 
    for xv in x_splits[1:-1]:
        ax.axvline(xv, color="black", lw=3)
    for yv in y_splits[1:-1]:
        ax.axhline(yv, color="black", lw=3)

    x0, x1 = x_splits[0], x_splits[-1]
    y0, y1 = y_splits[0], y_splits[-1]
    ax.add_patch(plt.Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        fill=False, edgecolor="black", lw=4, zorder=10
    ))

 
    letters = list(ABC[:16])  # A..P
    k = 0
    for r in range(len(y_splits) - 1):
        for c in range(len(x_splits) - 1):
            cx = 0.5 * (x_splits[c] + x_splits[c+1])
            cy = 0.5 * (y_splits[r] + y_splits[r+1])
            ax.text(
                cx, cy, letters[k],
                color="dimgray",
                ha="center", va="center",
                fontsize=36, alpha=0.8
            )
            k += 1
 
    def draw_arrows(points, color):
        for i in range(len(points) - 1):
            (x0, y0), (x1, y1) = points[i], points[i+1]
            ax.annotate(
                "",
                xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle=arrowstyle,
                    facecolor=color,
                    edgecolor=color,
                    lw=arrow_lw,
                    mutation_scale=arrow_scale,
                    shrinkA=0,
                    shrinkB=0,
                ),
                zorder=3,
            )
 
 
    draw_arrows(blue_xy,   color="#0b3c8c")  # bleu
    draw_arrows(purple_xy, color="#6a1b9a")  # violet
 
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Horizontal position (px)", fontsize=14)
    ax.set_ylabel("Vertical position (px)", fontsize=14)
    ax.tick_params(labelsize=11)

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()
    plt.clf()
    
    
def plot_scanpath_with_grid_arrows_circle(
    xlim=(0, 1200), ylim=(0, 800),
    x_splits=None, y_splits=None,
    save_path='figures/scanpath_grid.png',
    arrow_lw=1.6, arrow_scale=18, arrowstyle="-|>",
    circle_radii=[50, 70, 28, 22, 25, 52, 27, 34, 31, 48, 31],           # <-- rayon(s) des cercles
    circle_edgecolor="darkblue",               # contour des cercles
    circle_facecolor="darkblue",               # remplissage des cercles
    circle_alpha=0.20,                        # transparence du remplissage
):
    
    # Données (x, y)
    blue_xy = np.array([
       [995,760],[405,505],[560,640],[820,660],
       [1060,670],[915,425],[680,415],[342,220],[90,220], [265, 105], [405, 20]
    ], dtype=float)

    # Grille 4x4 par défaut
    if x_splits is None: x_splits = np.linspace(xlim[0], xlim[1], 5)
    if y_splits is None: y_splits = np.linspace(ylim[0], ylim[1], 5)

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots()

    # --- Grille noire ---
    for xv in x_splits[1:-1]: ax.axvline(xv, color="black", lw=3)
    for yv in y_splits[1:-1]: ax.axhline(yv, color="black", lw=3)
    x0, x1 = x_splits[0], x_splits[-1]
    y0, y1 = y_splits[0], y_splits[-1]
    ax.add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0,
                               fill=False, edgecolor="black", lw=4))

    # --- Lettres A–P ---
    letters = list(ABC[:16])
    k = 0
    for r in range(len(y_splits)-1):
        for c in range(len(x_splits)-1):
            cx = 0.5*(x_splits[c] + x_splits[c+1])
            cy = 0.5*(y_splits[r] + y_splits[r+1])
            ax.text(cx, cy, letters[k], color="dimgray",
                    ha="center", va="center", fontsize=36, alpha=0.8)
            k += 1

    # --- Flèches pleines entre points (aucun dot) ---
    def draw_arrows(points, color):
        for i in range(len(points) - 1):
            (x0, y0), (x1, y1) = points[i], points[i+1]
            ax.annotate(
                "", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle=arrowstyle,
                    facecolor=color, edgecolor=color,
                    lw=arrow_lw, mutation_scale=arrow_scale,
                    shrinkA=0, shrinkB=0
                ),
                zorder=3,
            )

    draw_arrows(blue_xy, color="#6a1b9a")  # violet

    # --- Cercles légèrement remplis sur les points ---
    # rayon : scalaire ou liste/array par point
    radii = np.asarray(circle_radii, dtype=float)
    if radii.size != len(blue_xy):
        raise ValueError("circle_radii doit avoir la même longueur que blue_xy.")

    for (x, y), r in zip(blue_xy, radii):
        circ = plt.Circle(
            (x, y), r,
            edgecolor=circle_edgecolor,
            facecolor=circle_facecolor,
            fill=True, alpha=circle_alpha,
            linewidth=1.2, zorder=4
        )
        ax.add_patch(circ)

    # --- Axes & mise en forme ---
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.invert_yaxis()                     # 0 en haut
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Horizontal position (px)", fontsize=14)
    ax.set_ylabel("Vertical position (px)", fontsize=14)
    ax.tick_params(labelsize=11)

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show(); plt.clf()
    
 
    
def dtw_figure_with_arrows():
    # --- Données ---
    blue_xy = np.array([
        [640, 420],
        [460, 610],
        [800, 630],
        [1040, 590],
        [985, 290],
        [798, 355],
        [580, 404],
        [597, 320],
        [605, 255],
        [655, 260],
        [703, 250],
        [1050, 85],
    ], dtype=float)

    purple_xy = np.array([
        [505, 403],
        [410, 560],
        [480, 535],
        [585, 515],
        [600, 580],
        [858, 430],
        [1020, 405],
        [1100, 290],
        [940, 190],
        [790, 295],
        [685, 395],
        [640, 315],
        [1055, 201],
    ], dtype=float)

    # --- DTW (coût cumulé + backtracking) ---
    def dtw(P, Q):
        n, m = len(P), len(Q)
        C = np.full((n, m), np.inf, float)
        parent = np.full((n, m, 2), -1, int)

        def d(i, j):
            return np.linalg.norm(P[i] - Q[j])

        C[0, 0] = d(0, 0)
        parent[0, 0] = [-1, -1]

        for i in range(1, n):
            C[i, 0] = C[i-1, 0] + d(i, 0)
            parent[i, 0] = [i-1, 0]
        for j in range(1, m):
            C[0, j] = C[0, j-1] + d(0, j)
            parent[0, j] = [0, j-1]

        for i in range(1, n):
            for j in range(1, m):
                cost = d(i, j)
                # insertion, suppression, match
                opts = [
                    (C[i-1, j],   (i-1, j)),
                    (C[i,   j-1], (i,   j-1)),
                    (C[i-1, j-1], (i-1, j-1)),
                ]
                prev_cost, prev_idx = min(opts, key=lambda t: t[0])
                C[i, j] = cost + prev_cost
                parent[i, j] = prev_idx

        # Chemin optimal
        path = []
        i, j = n-1, m-1
        while i >= 0 and j >= 0:
            path.append((i, j))
            if parent[i, j][0] == -1:
                break
            i, j = parent[i, j]
        path = path[::-1]

        # Plus grande distance locale le long du chemin (pour le segment rouge)
        local_dists = [np.linalg.norm(P[i] - Q[j]) for (i, j) in path]
        imax = int(np.argmax(local_dists))
        return C[n-1, m-1], path, imax

    dtw_cost, coupling, imax = dtw(blue_xy, purple_xy)

    # --- Figure ---
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots()

    # Flèches entre points (comme display_scanpath)
    def draw_arrows(points, color):
        for k in range(len(points) - 1):
            (x0, y0), (x1, y1) = points[k], points[k+1]
            ax.annotate(
                "",
                xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="-|>",
                    facecolor=color,
                    edgecolor=color,
                    lw=1.2,
                    mutation_scale=16,
                    shrinkA=0, shrinkB=0,
                ),
                zorder=3,
            )

    draw_arrows(blue_xy,   color="#0b3c8c")  # BLEU
    draw_arrows(purple_xy, color="#6a1b9a")  # VIOLET

    # Couplage DTW en pointillés
    for (i, j) in coupling:
        ax.plot([blue_xy[i, 0], purple_xy[j, 0]],
                [blue_xy[i, 1], purple_xy[j, 1]],
                "k--", lw=1, zorder=2)


    # Les annotations n'autoscalent pas -> fixer limites depuis les points
    all_pts = np.vstack([blue_xy, purple_xy])
    pad = 40
    ax.set_xlim(all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
    ax.set_ylim(all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)

    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()  # décommente si tu veux l'origine en haut
    ax.set_xlabel("Horizontal position (px)")
    ax.set_ylabel("Vertical position (px)") 
    
    save_path = 'figures/DTW.png'
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    
    plt.show()



def frechet_figure_with_arrows():
    # --- Données ---
    blue_xy = np.array([
        [640, 420],
        [460, 610],
        [800, 630],
        [1040, 590],
        [985, 290],
        [798, 355],
        [580, 404],
        [597, 320],
        [605, 255],
        [655, 260],
        [703, 250],
        [1050, 85],
    ], dtype=float)

    purple_xy = np.array([
        [505, 403],
        [410, 560],
        [480, 535],
        [585, 515],
        [600, 580],
        [858, 430],
        [1020, 405],
        [1100, 290],
        [940, 190],
        [790, 295],
        [685, 395],
        [640, 315],
        [1055, 201],
    ], dtype=float)

    # --- Fréchet discret (Eiter & Mannila) + backtracking ---
    def discrete_frechet(P, Q):
        n, m = len(P), len(Q)
        ca = np.full((n, m), np.inf, float)
        parent = np.full((n, m, 2), -1, int)

        def d(i, j):
            return np.linalg.norm(P[i] - Q[j])

        for i in range(n):
            for j in range(m):
                dist = d(i, j)
                if i == 0 and j == 0:
                    ca[i, j] = dist; parent[i, j] = [-1, -1]
                elif i > 0 and j == 0:
                    ca[i, 0] = max(ca[i-1, 0], dist); parent[i, 0] = [i-1, 0]
                elif i == 0 and j > 0:
                    ca[0, j] = max(ca[0, j-1], dist); parent[0, j] = [0, j-1]
                else:
                    # d[i,j] = max( dist, min(ca[i-1,j], ca[i-1,j-1], ca[i,j-1]) )
                    opts = [
                        (max(ca[i-1, j-1], dist), (i-1, j-1)),
                        (max(ca[i-1, j],   dist), (i-1, j)),  
                        (max(ca[i,   j-1], dist), (i,   j-1)),
                    ]
                    ca[i, j], parent[i, j] = min(opts, key=lambda t: t[0])

        # backtrack du couplage optimal
        path = []
        i, j = n-1, m-1
        while i >= 0 and j >= 0:
            path.append((i, j))
            if parent[i, j][0] == -1:
                break
            i, j = parent[i, j]
        path = path[::-1]

        # paire qui réalise la distance (max le long du chemin)
        dists = [np.linalg.norm(P[i] - Q[j]) for (i, j) in path]
        imax = int(np.argmax(dists))
        return ca[n-1, m-1], path, imax

    frechet_val, coupling, imax = discrete_frechet(blue_xy, purple_xy)

    # --- Figure ---
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots()

    # Flèches pleines entre points (comme display_scanpath)
    def draw_arrows(points, color):
        for k in range(len(points) - 1):
            (x0, y0), (x1, y1) = points[k], points[k+1]
            ax.annotate(
                "",
                xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="-|>",      # flèche pleine
                    facecolor=color,
                    edgecolor=color,
                    lw=1.2,
                    mutation_scale=16,
                    shrinkA=0, shrinkB=0,
                ),
                zorder=3,
            )

    draw_arrows(blue_xy,   color="#0b3c8c")  # BLEU
    draw_arrows(purple_xy, color="#6a1b9a")  # VIOLET

    # Couplage Fréchet optimal en pointillés
    for (i, j) in coupling:
        ax.plot([blue_xy[i, 0], purple_xy[j, 0]],
                [blue_xy[i, 1], purple_xy[j, 1]],
                "k--", lw=1, zorder=2)

    # Segment rouge = paire responsable de la distance de Fréchet
    i, j = coupling[imax]
    ax.plot([blue_xy[i, 0], purple_xy[j, 0]],
            [blue_xy[i, 1], purple_xy[j, 1]],
            color="red", lw=3, zorder=4)

    # Les annotations n'autoscalent pas -> fixer limites depuis les points
    all_pts = np.vstack([blue_xy, purple_xy])
    pad = 40
    ax.set_xlim(all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
    ax.set_ylim(all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)

    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()  # décommente si ton repère a l'origine en haut
    ax.set_xlabel("Horizontal position (px)")
    ax.set_ylabel("Vertical position (px)") 
    
    save_path = 'figures/frechet.png'
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    
    plt.show()

def art_4():
    
    data = 'dataset/nat006.csv' 
    image_ref = 'dataset/nat006.bmp'
    
    bs = v.BinarySegmentation(data, 
                              sampling_frequency = 500,  
                              segmentation_method = 'I_HMM',
                              distance_type = 'euclidean',                        
                              display_segmentation = False,
                              verbose=False,
                              size_plan_x = 921,
                              size_plan_y = 630,  
                              )
     
    v.AoISequence(bs, 
                  ref_image=image_ref,
                  AoI_identification_method='I_MS', 
                  display_AoI_path='figures/school',
                  verbose=False)
    

def AoI_scarf_plot_consensus(
    aoi_seqs,
    emine_res=None,
    sta_res=None,
    cdba_res=None,
    cdba_res2=None,   # <- NOUVEAU : 2e CDBA possible
    **kwargs
):
    show = kwargs.get("show", True)
    width_per_step = int(kwargs.get("width_per_step", 25))
    bar_width = float(kwargs.get("bar_width", 0.35))
    gap = float(kwargs.get("gap", 0.08))
    panel_gap = float(kwargs.get("panel_gap", 0.16))
    height_per_row = int(kwargs.get("height_per_row", 80))
    dest_ = kwargs.get("AoI_scarf_plot_path", None)
    title = kwargs.get("title", "")
    sta_key = kwargs.get("sta_key", "AoI_trend_analysis_common_subsequence")

    
    def _extract_seq(res, key):
        if res is None:
            return None
        if isinstance(res, dict):
            return res.get(key, None)
        return res

    def _rows_from_sequences(seq_rows):
        to_df = []
        max_len = 0
        for task, seq in seq_rows:
            seq = list(seq)
            max_len = max(max_len, len(seq))
            for k, lab in enumerate(seq):
                to_df.append(dict(Task=task, Start=k, Finish=k + 1, Resource=lab))
        df = pd.DataFrame(to_df)
        if df.empty:
            return df, 0
        # petit blanc entre bins : on réduit la largeur
        df["Finish"] = df["Start"] + (df["Finish"] - df["Start"]) * (1.0 - gap)
        return df, max_len

    def _make_gantt(df, colors):
        fig_local = ff.create_gantt(
            df,
            index_col="Resource",
            bar_width=bar_width,
            show_colorbar=True,
            group_tasks=True,
            colors=colors,
        )
        fig_local.update_layout(title_text="")  # remove "Gantt Chart"
        return fig_local

    def _dedupe_legend_traces(fig_local):
        seen = set()
        new_data = []
        for tr in fig_local.data:
            name = str(getattr(tr, "name", ""))
            if name in seen:
                tr.showlegend = False
            else:
                seen.add(name)
                tr.showlegend = True
            new_data.append(tr)
        fig_local.data = tuple(new_data)
        return fig_local

    def _force_alpha_legend(fig_local, ordered_labels):
        rank = {lab: i for i, lab in enumerate(ordered_labels)}
        for tr in fig_local.data:
            name = str(getattr(tr, "name", ""))
            if name in rank:
                tr.legendrank = rank[name]
        fig_local.update_layout(legend_traceorder="normal")
        return fig_local

    # -----------------------------
    # Build data (top / bottom)
    # -----------------------------
    # TOP: want 0 at top, last at bottom
    # create_gantt tends to put first category at bottom; easiest is to REVERSE row order here
    top_rows = [(str(i), list(aoi.sequence)) for i, aoi in enumerate(aoi_seqs)]
    top_rows_rev = list(reversed(top_rows))
    df_top, max_len_top = _rows_from_sequences(top_rows_rev)
    if df_top.empty:
        raise ValueError("Aucune séquence input à afficher (df_top vide).")

    # BOTTOM: sequences
    seq_emine = _extract_seq(emine_res, "AoI_eMine_common_subsequence")
    seq_sta = _extract_seq(sta_res, sta_key)

    # 1er CDBA (compatible ancien code)
    seq_cdba_1 = _extract_seq(cdba_res, "AoI_CDBA_common_subsequence")
    # 2e CDBA
    seq_cdba_2 = _extract_seq(cdba_res2, "AoI_CDBA_common_subsequence")

    # IMPORTANT:
    # - Pour afficher 2 lignes CDBA, on doit leur donner des noms internes distincts,
    #   sinon create_gantt les regroupe (group_tasks=True).
    CDBA1_TASK = "CDBA__1"
    CDBA2_TASK = "CDBA__2"

    # ordre désiré à l'affichage (haut -> bas)
    bottom_rows_desired = []
    if seq_emine is not None:
        bottom_rows_desired.append(("eMine", list(seq_emine)))
    if seq_sta is not None:
        bottom_rows_desired.append(("STA", list(seq_sta)))
    if seq_cdba_1 is not None:
        bottom_rows_desired.append((CDBA1_TASK, list(seq_cdba_1)))
    if seq_cdba_2 is not None:
        bottom_rows_desired.append((CDBA2_TASK, list(seq_cdba_2)))

    # IMPORTANT: ff.create_gantt met la 1ère task en bas -> on inverse pour avoir eMine en haut
    bottom_rows_plot = list(reversed(bottom_rows_desired))
    df_bot, max_len_bot = _rows_from_sequences(bottom_rows_plot)

    # impose l’ordre des Task pour éviter que Plotly réordonne
    if not df_bot.empty:
        categories_order = []
        if seq_emine is not None:
            categories_order.append("eMine")
        if seq_sta is not None:
            categories_order.append("STA")
        if seq_cdba_1 is not None:
            categories_order.append(CDBA1_TASK)
        if seq_cdba_2 is not None:
            categories_order.append(CDBA2_TASK)

        df_bot["Task"] = pd.Categorical(df_bot["Task"], categories=categories_order, ordered=True)
        df_bot = df_bot.sort_values(["Task", "Start"]).reset_index(drop=True)

    # -----------------------------
    # Shared color map (AoI letters)
    # -----------------------------
    if not df_bot.empty:
        all_resources = pd.concat([df_top["Resource"], df_bot["Resource"]], ignore_index=True)
    else:
        all_resources = df_top["Resource"]

    aoi_labels = sorted(all_resources.unique().tolist())
    colors_sns = sns.color_palette("pastel", n_colors=len(aoi_labels))
    colors = {lab: colors_sns[i] for i, lab in enumerate(aoi_labels)}

    # -----------------------------
    # Create individual gantts
    # -----------------------------
    fig_top = _make_gantt(df_top, colors)
    fig_bot = _make_gantt(df_bot, colors) if not df_bot.empty else None

    # capture tick mapping created by ff.create_gantt
    top_tickvals = list(fig_top.layout.yaxis.tickvals) if fig_top.layout.yaxis.tickvals else None
    top_ticktext = list(fig_top.layout.yaxis.ticktext) if fig_top.layout.yaxis.ticktext else None

    bot_tickvals, bot_ticktext = None, None
    if fig_bot is not None:
        bot_tickvals = list(fig_bot.layout.yaxis.tickvals) if fig_bot.layout.yaxis.tickvals else None
        bot_ticktext = list(fig_bot.layout.yaxis.ticktext) if fig_bot.layout.yaxis.ticktext else None

    # -----------------------------
    # Merge into subplots
    # -----------------------------
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=("", "")
    )

    for tr in fig_top.data:
        fig.add_trace(tr, row=1, col=1)
    if fig_bot is not None:
        for tr in fig_bot.data:
            fig.add_trace(tr, row=2, col=1)

    # copy shapes (bars) and remap to subplot axes
    shapes = []
    if fig_top.layout.shapes:
        for shp in fig_top.layout.shapes:
            s = shp.to_plotly_json()
            s["xref"] = "x"
            s["yref"] = "y"
            shapes.append(s)

    if fig_bot is not None and fig_bot.layout.shapes:
        for shp in fig_bot.layout.shapes:
            s = shp.to_plotly_json()
            s["xref"] = "x2"
            s["yref"] = "y2"
            shapes.append(s)

    fig.update_layout(shapes=shapes)

    # -----------------------------
    # Layout base
    # -----------------------------
    max_len = max(max_len_top, max_len_bot) if fig_bot is not None else max_len_top
    n_rows_total = len(top_rows) + (len(bottom_rows_desired) if bottom_rows_desired else 0)

    fig.update_layout(
        title_text=title,
        height=max(350, int(height_per_row * n_rows_total) + 120),
        width=max(520, int(max_len * width_per_step)),
        legend=dict(title=dict(text="AoI")),
        margin=dict(
    l=40,   # un peu de place pour les labels Y
    r=60,   # juste ce qu'il faut pour la légende
    t=40 if title else 25,
    b=35,
),
        template="plotly",
    )

    # remove any grids/axes
    fig.update_xaxes(showgrid=False, zeroline=False, showline=False, ticks="", showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showline=False, ticks="")

    # y titles
    fig.update_yaxes(title_text="AoI sequence index", row=1, col=1, autorange="reversed")
    if fig_bot is not None:
        fig.update_yaxes(title_text="Consensus sequence", row=2, col=1)

    # -----------------------------
    # Custom Y ticks
    # -----------------------------
    # TOP: displayed labels 0..N-1 from top to bottom.
    if top_tickvals is not None and top_ticktext is not None:
        map_top = {str(t): v for t, v in zip(top_ticktext, top_tickvals)}
        desired_top = [str(i) for i in range(len(aoi_seqs) - 1, -1, -1)]  # N-1 ... 0
        yvals_top = [map_top[l] for l in desired_top if l in map_top]

        fig.update_yaxes(
            row=1, col=1,
            tickmode="array",
            tickvals=yvals_top,
            ticktext=desired_top[:len(yvals_top)],
            autorange="reversed",
            ticks="",
        )

    # BOTTOM: mapping ticktext -> tickvals + affichage eMine / STA / CDBA / CDBA (sans distinction)
    if fig_bot is not None and bot_tickvals is not None and bot_ticktext is not None:
        map_bot = {str(t): v for t, v in zip(bot_ticktext, bot_tickvals)}

        # ordre interne (noms uniques)
        desired_internal = []
        desired_display = []
        if seq_emine is not None:
            desired_internal.append("eMine")
            desired_display.append("eMine")
        if seq_sta is not None:
            desired_internal.append("STA")
            desired_display.append("STA")
        if seq_cdba_1 is not None:
            desired_internal.append(CDBA1_TASK)
            desired_display.append("CDBA")
        if seq_cdba_2 is not None:
            desired_internal.append(CDBA2_TASK)
            desired_display.append("CDBA")

        yvals_use = [map_bot[l] for l in desired_internal if l in map_bot]

        fig.update_yaxes(
            row=2, col=1,
            tickmode="array",
            tickvals=yvals_use,
            ticktext=desired_display[:len(yvals_use)],
            ticks="",
        )

    # -----------------------------
    # TRUE GAP between panels: equal-size panels
    # -----------------------------
    g = panel_gap
    mid = 0.5
    fig.update_layout(
        yaxis=dict(domain=[mid + g / 2, 1.0]),
        yaxis2=dict(domain=[0.0, mid - g / 2]),
    )

    # -----------------------------
    # Legend: dedupe + alphabetical
    # -----------------------------
    _dedupe_legend_traces(fig)
    _force_alpha_legend(fig, aoi_labels)

    # show/save
    if show:
        fig.show()
    if dest_ is not None:
        fig.write_image(dest_ + "consensus.png", scale =4)

    return fig

 
import plotly.graph_objects as go 

  

def plot_ref_seq_with_bi_tri_grams(
    aoi_seq,
    bigram_counts,
    trigram_counts,
    k=8,
    title="",
    width_per_step=25,
    height=520,
    gap=0.12,
    scarf_thickness=0.28,
    show=True,
    horizontal_spacing=0.22,
    dest_=None,
    scarf_bg_color="rgb(231,238,246)",   # bleu clair UNIQUEMENT pour la frise
):
    # ---------- helpers ----------
    def _clean_ngram_key(s):
        s = str(s).strip()
        if s.endswith(","):
            s = s[:-1]
        s = ",".join([p.strip() for p in s.split(",") if p.strip() != ""])
        return s

    def _topk(counter_like, k_):
        items = list(counter_like.items())
        items.sort(key=lambda x: float(x[1]), reverse=True)
        items = items[:k_]
        labels = [_clean_ngram_key(x[0]) for x in items]
        vals = [float(x[1]) for x in items]
        return labels, vals

    def _rgb(c):
        r, g, b = [int(round(255 * x)) for x in c]
        return f"rgb({r},{g},{b})"

    # ---------- data ----------
    aoi_seq = list(aoi_seq)
    n = len(aoi_seq)

    bi_labels, bi_vals = _topk(bigram_counts, k)
    tri_labels, tri_vals = _topk(trigram_counts, k)

    # Pastel blue/red for bottom bars
    pal_bar = sns.color_palette("pastel")
    blue_bar = _rgb(pal_bar[0])  # bleu pastel
    red_bar  = _rgb(pal_bar[3])  # rouge pastel

    # AoI colors (pastel)
    aoi_labels = sorted(pd.unique(pd.Series(aoi_seq)).tolist())
    pal_aoi = sns.color_palette("pastel", n_colors=len(aoi_labels))
    color_map = {lab: pal_aoi[i] for i, lab in enumerate(aoi_labels)}

    # ---------- figure ----------
    fig = make_subplots(
        rows=2, cols=2,
        shared_xaxes=False,
        row_heights=[0.40, 0.60],
        vertical_spacing=0.10,
        horizontal_spacing=horizontal_spacing,
        specs=[
            [{"colspan": 2}, None],
            [{}, {}]
        ],
        subplot_titles=("", "Top bi-grams", "Top tri-grams")
    )

    # --- Row 1: scarf (shapes) ---
    fig.add_trace(
        go.Scatter(x=[0, n], y=[0, 0], mode="markers", marker_opacity=0, showlegend=False),
        row=1, col=1
    )

    shapes = []

    # scarf rectangles
    y_mid = 0.5
    half_t = scarf_thickness / 2
    y0, y1 = y_mid - half_t, y_mid + half_t
    w = 1.0 - gap

    for i, lab in enumerate(aoi_seq):
        shapes.append(dict(
            type="rect",
            xref="x", yref="y",
            x0=i, x1=i + w,
            y0=y0, y1=y1,
            fillcolor=_rgb(color_map.get(lab, pal_aoi[0])),
            line=dict(width=0),
            layer="above"  # au-dessus du fond bleu, mais sous la légende
        ))

    fig.update_layout(shapes=shapes)

    fig.update_xaxes(row=1, col=1, range=[0, n], showgrid=False, zeroline=False, ticks="", showticklabels=False)
    fig.update_yaxes(row=1, col=1, range=[0, 1], showgrid=False, zeroline=False, ticks="", showticklabels=False)

    # --- Input sequence (hors fond bleu : on le met dans la marge gauche) ---
    fig.add_annotation(
        text="Input sequence",
        xref="paper", yref="paper",
        x=-0.06, y=0.78,
        xanchor="center",
        yanchor="middle",
        textangle=-90,
        showarrow=False,
        font=dict(size=16),
    )

    # --- Legend AoI: lignes (pas de points) ---
    for rank, lab in enumerate(aoi_labels):
        fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[None, None],
                mode="lines",
                line=dict(color=_rgb(color_map[lab]), width=10),
                name=str(lab),
                showlegend=True,
                legendrank=rank,
                hoverinfo="skip",
            ),
            row=1, col=1
        )

    # --- Row 2: bi-grams ---
    fig.add_trace(
        go.Bar(
            x=bi_vals[::-1],
            y=bi_labels[::-1],
            width=0.5,
            orientation="h",
            text=[f"{v:.2%}" for v in bi_vals[::-1]],
            textposition="outside",
            cliponaxis=False,
            marker=dict(color='darkblue'),
            showlegend=False
        ),
        row=2, col=1
    )

    # --- Row 2: tri-grams ---
    fig.add_trace(
        go.Bar(
            x=tri_vals[::-1],
            y=tri_labels[::-1],
            width=0.5,
            orientation="h",
            text=[f"{v:.2%}" for v in tri_vals[::-1]],
            textposition="outside",
            cliponaxis=False,
            marker=dict(color='darkblue'),
            showlegend=False
        ),
        row=2, col=2
    )
    
    fig.update_xaxes(
    row=2, col=1,
    showticklabels=False,
    ticks="",
    showgrid=False
    )
    
    fig.update_xaxes(
        row=2, col=2,
        showticklabels=False,
        ticks="",
        showgrid=False
    )

    # axes bas
    fig.update_xaxes(row=2, col=1, tickformat=".0%", showgrid=True, zeroline=False)
    fig.update_xaxes(row=2, col=2, tickformat=".0%", showgrid=True, zeroline=False)
    fig.update_yaxes(row=2, col=1, automargin=True)
    fig.update_yaxes(row=2, col=2, automargin=True)

    # --- Layout global : fond blanc partout ---
    fig.update_layout(
        title=title,
        height=height,
        width=max(740, int(n * width_per_step)),
        template="plotly",
        paper_bgcolor="white",  # <-- fond global blanc
        plot_bgcolor="white",   # <-- plots bas en blanc
        margin=dict(l=110, r=140, t=70 if title else 40, b=55),
        legend=dict(
            title=dict(text="AoI"),
            traceorder="normal",
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.05,  # dans la marge droite -> hors rectangle bleu
        ),
    )

    # --- Fond bleu UNIQUEMENT derrière la frise (subplot du haut) ---
    # On récupère le domain du subplot du haut (xaxis/yaxis) en coordonnées paper
    xdom = fig.layout.xaxis.domain
    ydom = fig.layout.yaxis.domain
    
    pad_x = 0.04  # augmente si besoin (0.02 à 0.06)
    
    fig.add_shape(
        type="rect",
        xref="paper", yref="paper",
        x0=xdom[0] - pad_x,   # <-- peut être négatif
        x1=xdom[1] + pad_x,   # <-- peut être > 1
        y0=ydom[0],
        y1=ydom[1],
        fillcolor=scarf_bg_color,
        line=dict(width=0),
        layer="below",
    )
    
    if dest_ is not None:
        fig.write_image(dest_ + "ngrams.png", scale=4)
    if show:
        fig.show()

    return fig




# root = 'dataset/'
# sp1 = v.Scanpath(root + 'data_1.csv', 
#                 sampling_frequency = 256,                  
#                 segmentation_method = 'I_HMM',
#                 distance_type = 'angular',                        
#                 display_segmentation = False,
#                 size_plan_x = 1200,
#                 size_plan_y = 800, 
#                 display_scanpath=True,
#                 verbose=False)

# sp2 = v.Scanpath(root + 'data_2.csv', 
#                 sampling_frequency = 256,  
#                 segmentation_method = 'I_HMM',
#                 distance_type = 'angular',                        
#                 display_segmentation = False,
#                 size_plan_x = 1200,
#                 size_plan_y = 800,
#                 display_scanpath=True,
#                 verbose=True)

# sp3 = v.Scanpath(root + 'data_3.csv', 
#                 sampling_frequency = 256,                  
#                 segmentation_method = 'I_HMM',
#                 distance_type = 'angular',                        
#                 display_segmentation = False,
#                 size_plan_x = 1200,
#                 size_plan_y = 800,
#                 display_scanpath=True,
#                 verbose=False)

# sp4 = v.Scanpath(root + 'data_4.csv', 
#                 sampling_frequency = 256,                  
#                 segmentation_method = 'I_HMM',
#                 distance_type = 'angular',                        
#                 display_segmentation = False,
#                 size_plan_x = 1200,
#                 size_plan_y = 800,
#                 display_scanpath=True,
#                 verbose=False)

# sp5 = v.Scanpath(root + 'data_5.csv', 
#                 sampling_frequency = 256,                  
#                 segmentation_method = 'I_HMM',
#                 distance_type = 'angular',                        
#                 display_segmentation = False,
#                 size_plan_x = 1200,
#                 size_plan_y = 800,
#                 display_scanpath=True,
#                 verbose=False)

# sp6 = v.Scanpath(root + 'data_6.csv', 
#                 sampling_frequency = 256,                  
#                 segmentation_method = 'I_HMM',
#                 distance_type = 'angular',                        
#                 display_segmentation = False,
#                 size_plan_x = 1200,
#                 size_plan_y = 800,
#                 display_scanpath=True,
#                 verbose=False)

# seqs = [sp1, sp4, sp5, sp6] 
# aoi_seqs = v.AoI_sequences(
#     seqs,
#     display_scanpath=True,
#     AoI_identification_method="I_KM",
#     AoI_IKM_cluster_number=5,
# )

# for seq in aoi_seqs:
#     print(repr(seq.sequence)
#           )
#     print(repr(seq.durations)
#           )
#     print(repr(seq.centers)
#           )
#     print()
    
# v.AoI_scarf_plot(aoi_seqs)

# emine = v.AoI_eMine(aoi_seqs)
# cdba_1  = v.AoI_CDBA(aoi_seqs,
#                     AoI_CDBA_initial_random_state=3)
# cdba_2  = v.AoI_CDBA(aoi_seqs,
#                     AoI_CDBA_initial_random_state=5)
# sta = v.AoI_trend_analysis(aoi_seqs,
#                             AoI_trend_analysis_tolerance_level=0.5)
 
# AoI_scarf_plot_consensus(
#     aoi_seqs,
#     emine_res=emine,
#     sta_res=sta,
#     cdba_res=cdba_1, 
#     AoI_scarf_plot_path='figures/',  # ou "path/to/output/"
# )

# AoI1 = v.AoISequence(sp5,
#                       AoI_identification_method="I_KM",
#                       AoI_IKM_cluster_number=4,
#                       verbose=False,
#                       AoI_temporal_binning='collapse')
# print(AoI1.sequence,)
# n2=(v.AoI_NGram(AoI1,
#                   verbose=False,
#                   AoI_NGram_length=2))
# n3=(v.AoI_NGram(AoI1,
#                   verbose=False,
#                   AoI_NGram_length=3))



# fig = plot_ref_seq_with_bi_tri_grams(
#     AoI1.sequence,
#     bigram_counts=n2['AoI_NGram'],
#     trigram_counts=n3['AoI_NGram'],
#     k=8, 
#     dest_='figures/'
# )

centers = {
  "A": np.array([720.0, 260.0]),
  "B": np.array([360.0, 210.0]),
  "C": np.array([980.0, 330.0]),
  "D": np.array([760.0, 650.0]),
  "E": np.array([420.0, 620.0]),
}

aoi1 = v.AoISequence(dict({'sequence': ['A','A','A','B','B','B','C','C','A','A','E','E','A','A','C','C','C',
                                                           'B','B','A','A','A','D','D','E','B','B','C','A','A','A','C','C','D','E','A'] ,
                                              'durations': np.array([
                                                  0.14,0.20,0.17,0.12,0.31,0.25,0.35,0.21,0.18,0.50,
                                                  0.13,0.21,0.19,0.13,0.16,0.21,0.93,0.14,0.18,0.73,
                                                  0.21,0.14,0.12,0.30,0.22,0.16,0.20,0.18,0.11,0.25,
                                                  0.18,0.12,0.21,0.18,0.13,0.15,0.22,0.35
                                                      ]),
                                              'centers': centers}))

aoi2 = v.AoISequence(dict({'sequence': ['A','A','B','B','B','C','C','C','A','E','E','A','A','A','C','C',
                                            'B','B','B','D','D','E','E','B','A','A','C','C','C','B','B',
                                            'A','A','A','C','C','D','B','B','A','A'],
                                              'durations': np.array([
                                                  0.16,0.14,0.14,0.36,0.20,0.31,0.24,0.18,0.51,0.12,
                                                  0.21,0.19,0.15,0.56,0.14,0.21,0.13,0.41,0.20,0.11,
                                                  0.28,0.23,0.15,0.20,0.87,0.20,0.17,0.13,0.50,0.12,
                                                  0.20,0.14,0.21,0.16,0.25,0.20,0.13,0.24,0.18,0.35,
                                                  0.22,0.31
                                                ]),
                                              'centers': centers}))

aoi3 = v.AoISequence(dict({'sequence': ['A','A','A','C','C','B','B','B','A','A','E','E','D','D','A','A',
                                                'C','C','B','B','E','A','A','A','C','C','D','E','B','B','A','A','C','C','A'],
                                              'durations': np.array([
                                                  0.15,0.21,0.18,0.20,0.16,0.13,0.27,0.20,0.56,0.18,
                                                  0.12,0.22,0.11,0.30,0.19,0.14,0.15,0.21,0.14,0.41,
                                                  0.19,0.87,0.23,0.15,0.18,0.13,0.74,0.12,0.15,0.16,
                                                  0.20,0.18,0.50,0.20,0.35
                                                ]),
                                              'centers': centers}))

aoi4 = v.AoISequence(dict({'sequence': ['B','B','B','A','A','C','C','C','B','B','A','A','E','E','A','C','C',
                                                'B','B','B','D','D','E','B','A','A','A','C','C','E','B','B','A','A',
                                                'A','C','D','E','A','A','A','B','B','C','C','A','A'],
                                              'durations': np.array([
                                                              0.13,0.39,0.20,0.14,0.51,0.21,0.16,0.74,0.14,0.35,
                                                              0.21,0.16,0.12,0.21,0.87,0.15,0.19,0.13,0.41,0.20,
                                                              0.11,0.30,0.22,0.15,0.73,0.20,0.14,0.25,0.20,0.14,
                                                              0.16,0.18,0.93,0.21,0.15,0.21,0.13,0.15,0.27,0.35,
                                                              0.18,0.22,0.14,0.19,0.16,0.24,0.31
                                                            ]),
                                              'centers': centers}))

aoi_seqs = [aoi1,aoi2, aoi3, aoi4]


emine = v.AoI_eMine(aoi_seqs)


sta = v.AoI_trend_analysis(aoi_seqs,
                            AoI_trend_analysis_tolerance_level=0.5)

cdba_1  = v.AoI_CDBA(aoi_seqs)

print(emine)
print(sta)
print(cdba_1)

AoI_scarf_plot_consensus(
    aoi_seqs,
    emine_res=emine,
    sta_res=sta,
    cdba_res=cdba_1, 
    AoI_scarf_plot_path='figures/',  # ou "path/to/output/"
)




