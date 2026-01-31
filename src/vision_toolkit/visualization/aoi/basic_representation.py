# -*- coding: utf-8 -*-

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

np.random.seed(1)


def display_aoi_predefined_reference_image(positions, clusters, config, ref_image):
    import cv2
   
    path = config.get("display_AoI_path", None)

    # Load image if a path is provided
    if isinstance(ref_image, str):
        ref_image = cv2.imread(ref_image, cv2.IMREAD_COLOR)
        if ref_image is None:
            raise FileNotFoundError(f"Could not read image at: {ref_image}")
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

    # Resize to configured display size
    ref_image = cv2.resize(ref_image, (config["size_plan_x"], config["size_plan_y"]))

    plt.style.use("seaborn-v0_8")

    fig, ax = plt.subplots()
    ax.imshow(ref_image, alpha=0.4)
    ax.grid(False)

    # More robust than "pastel" on top of a photo
    colors_sns = sns.color_palette("colorblind", n_colors=len(clusters.keys()))

    aoi_coords = np.array(config["AoI_coordinates"])

    for i, k_ in enumerate(sorted(clusters.keys())):
        idx = clusters[k_]

        aoi_coord = aoi_coords[i]
        xy = (aoi_coord[0, 0], aoi_coord[0, 1])
        w_ = aoi_coord[1, 1] - aoi_coord[0, 1]
        h_ = aoi_coord[1, 0] - aoi_coord[0, 0]

        # Rectangle: keep semi-transparent fill but add a strong edge
        rect = patches.Rectangle(
            xy,
            h_,
            w_,
            linewidth=2,
            edgecolor=colors_sns[i],
            facecolor=colors_sns[i],
            alpha=0.25,   # slightly lower to avoid washing out the photo
            fill=True,
            zorder=2,
        )
        ax.add_patch(rect)

        
        ax.scatter(
            positions[0, idx],
            positions[1, idx],
            color=colors_sns[i],
            marker="o",
            s=28,
            edgecolor="black",
            linewidth=0.7,
            zorder=3,
        )
 
        x_m = (aoi_coord[1, 0] + aoi_coord[0, 0]) / 2
        y_m = (aoi_coord[1, 1] + aoi_coord[0, 1]) / 2

        ax.text(
            x_m,
            y_m,
            str(k_),
            fontsize=15,
            ha="center",
            va="center",
            color="black",
            zorder=4,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.7),
        )

    ax.set_xlabel("Horizontal position (px)", fontsize=12)
    ax.set_ylabel("Vertical position (px)", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)

    ax.set_xlim(0, config["size_plan_x"])
    ax.set_ylim(0, config["size_plan_y"])
    ax.invert_yaxis()

    if path is not None:
        out_path = path + "_AoI_reference_image.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")

    plt.show()
    plt.clf()



def display_aoi_identification_reference_image(positions, clusters, config, ref_image):
    import cv2
 
    path = config.get("display_AoI_path", None)

    # Load image if a path is provided
    if isinstance(ref_image, str):
        ref_image = cv2.imread(ref_image, cv2.IMREAD_COLOR)
        if ref_image is None:
            raise FileNotFoundError(f"Could not read image at: {ref_image}")
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

    # Resize to configured display size
    ref_image = cv2.resize(ref_image, (config["size_plan_x"], config["size_plan_y"]))

    # Create figure
    fig, ax = plt.subplots()

    # Show reference image
    ax.imshow(ref_image, alpha=0.4)
    ax.grid(False)

    # Color palette (kept as in your discussion)
    colors = sns.color_palette("colorblind", n_colors=len(clusters.keys()))

    # Plot clusters
    for i, k_ in enumerate(sorted(clusters.keys())):
        idx = clusters[k_]

        # --- Solution 1: add a black edge around points for visibility ---
        ax.scatter(
            positions[0, idx],
            positions[1, idx],
            color=colors[i],
            s=45,
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )

        # Single-point cluster: draw a tiny dashed circle + label
        if len(idx) == 1:
            x_m = float(positions[0, idx][0])
            y_m = float(positions[1, idx][0])

            circle = plt.Circle(
                (x_m, y_m),
                1e-6,
                color="black",
                linewidth=2,
                linestyle="--",
                fill=False,
                zorder=4,
            )
            ax.add_patch(circle)

            ax.text(
                x_m,
                y_m,
                str(k_),
                fontsize=25,
                color="black",
                ha="center",
                va="center",
                zorder=5,
            )
        else:
            # Requires your existing helper
            plot_confidence_ellipse(
                positions[:, idx],
                name=k_,
                ax=ax,
                color="black",
            )

    # Axes formatting
    ax.set_xlabel("Horizontal position (px)", fontsize=12)
    ax.set_ylabel("Vertical position (px)", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)

    ax.set_xlim(0, config["size_plan_x"])
    ax.set_ylim(0, config["size_plan_y"])
    ax.invert_yaxis()

    # Save if requested
    if path is not None:
        # Keep your original naming; add extension for clarity
        out_path = path + "_AoI_reference_image.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")

    plt.show()
    plt.clf()



def display_aoi_identification(positions, clusters, config):
 
    path = config.get("display_AoI_path", None)

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots()

    # More robust than "pastel"
    colors_sns = sns.color_palette("colorblind", n_colors=len(clusters.keys()))

    for i, k_ in enumerate(sorted(clusters.keys())):
        idx = clusters[k_]

        # --- Solution 1: black outline for visibility ---
        ax.scatter(
            positions[0, idx],
            positions[1, idx],
            color=colors_sns[i],
            s=35,
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )

        if len(idx) == 1:
            x_m = float(positions[0, idx][0])
            y_m = float(positions[1, idx][0])

            circle = plt.Circle(
                (x_m, y_m),
                1e-6,
                color="black",
                linewidth=2,
                linestyle="--",
                fill=False,
                zorder=4,
            )
            ax.add_patch(circle)

            ax.text(
                x_m,
                y_m,
                str(k_),
                fontsize=22,
                color="black",
                ha="center",
                va="center",
                zorder=5,
            )
        else:
            # Uses your existing helper
            plot_confidence_ellipse(
                positions[:, idx],
                name=k_,
                ax=ax,
                color="black",
            )

    ax.set_xlabel("Horizontal position (px)", fontsize=12)
    ax.set_ylabel("Vertical position (px)", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)

    ax.set_xlim(0, config["size_plan_x"])
    ax.set_ylim(0, config["size_plan_y"])
    ax.invert_yaxis()

    if path is not None:
        fig.savefig(path + "_AoI.png", dpi=200, bbox_inches="tight")

    plt.show()
    plt.clf()



def plot_confidence_ellipse(
    positions,
    name,
    ax,
    color,
    p=0.68,
):
 
    plt.style.use("seaborn-v0_8")

    cov = np.cov(positions[0], positions[1])

    eigenvalues, eigenvectors = np.linalg.eig(cov)

    i = np.argmax(eigenvalues)
    i_ = np.argmin(eigenvalues)

    ei = eigenvalues[i]
    ei_ = eigenvalues[i_]

    ev = eigenvectors[:, i]
    angle = np.arctan2(ev[1], ev[0])

    if angle < 0:
        angle += 2 * np.pi

    x_m = np.mean(positions[0])
    y_m = np.mean(positions[1])

    chisquare_val = stats.chi2.ppf(p, df=2)

    a = np.sqrt(chisquare_val * ei)
    b = np.sqrt(chisquare_val * ei_)

    theta_grid = np.linspace(0, 2 * np.pi)

    ellipse_x_r = a * np.cos(theta_grid)
    ellipse_y_r = b * np.sin(theta_grid)

    rot_mat = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )

    r_ellipse = np.matmul(rot_mat, np.vstack((ellipse_x_r, ellipse_y_r)))

    ax.plot(
        r_ellipse[0] + x_m, r_ellipse[1] + y_m, color=color, linewidth=2, linestyle="--"
    )

    ax.text(x_m, y_m, name, fontsize=22)
