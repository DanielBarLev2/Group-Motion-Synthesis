import os
import numpy as np
import matplotlib.pyplot as plt

import colorsys
import matplotlib
import matplotlib.colors as mc
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from integration.config_files import cfg


def plot_world_trajectories(trajectories: np.ndarray,
                            ax: plt.Axes = None,
                            show: bool = True,
                            output_filename: str = None) -> None:
    """
    Plots 2D trajectories in world coordinates.
    Each trajectory is plotted with a different color.
    Args:
        trajectories: Array of world positions, where each element represents a trajectory.
        ax: Matplotlib axis to draw the trajectory. Creates a new axis if None.
        show: Whether to display the plot using plt.show().
        output_filename: Path to save the plot image. Defaults to save in the current working directory.

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each trajectory in different colors
    for index in range(trajectories.shape[0]):
        ax.plot(trajectories[index, :, 0], trajectories[index, :, 1], marker='o', linestyle='-', 
                label=f"{index+1}")

    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_title("2D Trajectories Plot")
    ax.legend(loc='best')

    if output_filename is None:
        output_path = os.path.join(cfg.OUTPUTS_DIR, "plot.png")
    else:
        if not os.path.isabs(output_filename):
            output_path = os.path.join(cfg.OUTPUTS_DIR, output_filename)

    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"plot saved to {output_path}")

    if show:
        plt.show()


def darken_color(color, amount=0.8):
    """
    Darkens the given color in HLS color space.
    amount=1.0 returns the same color; >1.0 => more dark, <1.0 => lighter.
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    r, g, b = mc.to_rgb(c)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0, min(1, l * amount))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (r, g, b)


def lighten_color(color, amount=1.25):
    """
    Lightens the given color in HLS color space.
    amount=1.0 => same color; >1.0 => lighter, <1.0 => darker.
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    r, g, b = mc.to_rgb(c)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0, min(1, l * amount))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (r, g, b)


def plot_multi_3d_motion_extended(
    save_path,
    kinematic_tree,
    motions_list,
    init_coordinations,
    dataset='humanml',
    fps=120,
    color_mode='multi',
    show_trajectories=True,
    titles=None,
    camera_view='default'):
    """
    Create one MP4 showing multiple motions (skeletons) in the same 3D world, with:
      - Grey floor at y=0
      - Skeleton root traces on the floor (y=0) if show_trajectories=True
      - Automatic bounding for x,z
      - Various camera_view options:
          * 'default': elevated ~120°, azim=-90
          * 'top':     top-down elev=90,  azim=-90
          * 'side':    side view elev=0,  azim=-90
          * 'side2':   another side elev=0, azim=0
      - Single or combined figure title
    """

    matplotlib.use('Agg')

    num_skeletons = len(motions_list)
    if init_coordinations.shape[0] != num_skeletons:
        raise ValueError("init_coordinations must have exactly one row per skeleton!")

    # --- 1) Convert each motion to (T, J, 3), apply dataset scaling ---
    data_list = []
    max_frames = 0

    for i, motion in enumerate(motions_list):
        # Ensure shape is (T, J, 3)
        if motion.ndim == 2:
            T, D = motion.shape
            J = D // 3
            motion = motion.reshape(T, J, 3)

        T = motion.shape[0]
        max_frames = max(max_frames, T)

        # Scale
        if dataset == "kit":
            motion = motion * 0.003
        elif dataset == "humanml":
            motion = motion * 1.3
        elif dataset in ["humanact12", "uestc"]:
            motion = motion * -1.5
        elif dataset == "babel":
            motion = motion * 1.3

        data_list.append(motion)

    # --- 2) Apply init_coordinations offsets ---
    offset_data_list = []
    for i, data in enumerate(data_list):
        skeleton_data = data.copy()
        skeleton_data[..., 0] += init_coordinations[i, 0]
        skeleton_data[..., 1] += init_coordinations[i, 1]
        skeleton_data[..., 2] += init_coordinations[i, 2]
        offset_data_list.append(skeleton_data)

    # --- 3) Compute bounding box in x,z (after offsets) ---
    all_concat = [sk.reshape(-1, 3) for sk in offset_data_list]
    all_concat = np.concatenate(all_concat, axis=0)

    min_vals = np.min(all_concat, axis=0)  # [min_x, min_y, min_z]
    max_vals = np.max(all_concat, axis=0)  # [max_x, max_y, max_z]

    # Shift everything up so floor is at y=0
    lowest_y = min_vals[1]
    if lowest_y < 0:
        for sk in offset_data_list:
            sk[..., 1] -= lowest_y
        # re-calc bounding box
        all_concat = [sk.reshape(-1, 3) for sk in offset_data_list]
        all_concat = np.concatenate(all_concat, axis=0)
        min_vals = np.min(all_concat, axis=0)
        max_vals = np.max(all_concat, axis=0)

    floor_y = 0.0  # we fix the floor at y=0

    # --- bounding box for x,z ---
    x_min, x_max = min_vals[0], max_vals[0]
    z_min, z_max = min_vals[2], max_vals[2]
    size_x = x_max - x_min
    size_z = z_max - z_min
    max_dim = max(size_x, size_z)
    margin = 0.1 * max_dim if max_dim > 0 else 1.0

    x_min -= margin
    x_max += margin
    z_min -= margin
    z_max += margin

    y_min = 0
    y_max = max_vals[1] + margin

    # --- 4) Create figure/axes ---
    fig = plt.figure(figsize=(6, 6))
    ax = Axes3D(fig)

    # set bounding box
    ax.set_xlim3d([x_min, x_max])
    ax.set_ylim3d([y_min, y_max])
    ax.set_zlim3d([z_min, z_max])

    # set camera angles
    if camera_view == 'top':
        ax.view_init(elev=90, azim=-90)
    elif camera_view == 'side':
        ax.view_init(elev=0, azim=-90)
    elif camera_view == 'side2':
        ax.view_init(elev=0, azim=0)
    else:  # 'default'
        ax.view_init(elev=120, azim=-90)

    ax.dist = 7.0 + 0.02 * max_dim  # simple heuristic

    # Titles
    if isinstance(titles, str):
    # Omit or handle as needed, e.g.:
        pass  # do nothing
    elif isinstance(titles, list) and len(titles) > 0:
        # Or handle a short version, e.g.:
        short_title = "Multiple Motions"
        fig.suptitle(short_title, fontsize=8)

    # Turn off axis ticks
    plt.axis("off")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # --- 5) Prepare colors: skeleton vs. trajectory (on floor) ---
    color_candidates = [
        "#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231", 
        "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabebe",
        "#469990", "#dcbeff"
    ]
    while len(color_candidates) < num_skeletons:
        color_candidates += color_candidates

    if color_mode == 'multi':
        base_colors = color_candidates[:num_skeletons]
    else:
        # same color for all
        base_colors = ["#ff6d00"] * num_skeletons

    skeleton_colors = [darken_color(c, 1.2) for c in base_colors]   # darker
    trace_colors    = [lighten_color(c, 1.4) for c in base_colors]  # lighter

    # build root positions
    root_positions = [sk[:, 0, :] for sk in offset_data_list]  # each shape (T, 3)

    # --- 6) floor drawing ---
    def plot_xz_floor(ax, floorY, x_min, x_max, z_min, z_max):
        verts = [
            [x_min, floorY, z_min],
            [x_min, floorY, z_max],
            [x_max, floorY, z_max],
            [x_max, floorY, z_min],
        ]
        plane = Poly3DCollection([verts])
        plane.set_facecolor((0.5, 0.5, 0.5, 0.4))  # grey, semi-transparent
        ax.add_collection3d(plane)

    # --- 7) update function for FuncAnimation ---
    def update(frame_idx):
        ax.lines = []
        ax.collections = []
        plot_xz_floor(ax, floor_y, x_min, x_max, z_min, z_max)

        for sk_idx, data in enumerate(offset_data_list):
            T = data.shape[0]
            use_frame = min(frame_idx, T - 1)
            coords_frame = data[use_frame]  # shape (J, 3)

            # draw skeleton
            for chain in kinematic_tree:
                chain_coords = coords_frame[chain]
                ax.plot3D(
                    chain_coords[:, 0],
                    chain_coords[:, 1],
                    chain_coords[:, 2],
                    linewidth=4.0 if chain == kinematic_tree[0] else 2.0,
                    color=skeleton_colors[sk_idx]
                )

            # if show_trajectories => plot on the floor at y=0
            # the root's x,z, but fix y=0
            if show_trajectories:
                root_slice = root_positions[sk_idx][:use_frame+1]
                if len(root_slice) > 1:
                    # root_slice[:, 0] => x
                    # root_slice[:, 2] => z
                    # fix y=0
                    ax.plot3D(
                        root_slice[:, 0],
                        [floor_y]*len(root_slice),
                        root_slice[:, 2],
                        linewidth=2.0,
                        color=trace_colors[sk_idx]
                    )

    ani = FuncAnimation(fig, update, frames=max_frames, interval=1000//fps, repeat=False)
    ani.save(save_path, fps=fps)
    plt.close(fig)
    print(f"Animation saved to {save_path}")