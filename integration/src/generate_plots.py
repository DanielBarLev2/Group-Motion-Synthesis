import os
import numpy as np
import matplotlib.pyplot as plt

from integration.config_files import cfg


def plot_world_trajectories(trajectories: np.ndarray,
                            ax: plt.Axes = None,
                            show: bool = True,
                            output_filename: str = None) -> None:
    """
    Plots 2D trajectories in world coordinates, mirrored relative to the X-axis.
    Each trajectory is plotted with a different color, and the plot is square.

    Args:
        trajectories: Array of world positions, where each element represents a trajectory.
        ax: Matplotlib axis to draw the trajectory. Creates a new axis if None.
        show: Whether to display the plot using plt.show().
        output_filename: Path to save the plot image. Defaults to save in the current working directory.

    Returns:
        None
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each trajectory in different colors, mirroring relative to X-axis
    for index in range(trajectories.shape[0]):
        ax.plot(trajectories[index, :, 0], -trajectories[index, :, 1], marker='o', linestyle='-', 
                label=f"Trajectory {index+1}")

    # Set labels, title, and legend
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_title("2D Trajectories Plot")
    ax.legend(loc='best')

    # Ensure the plot has equal aspect ratio for square visualization
    ax.set_aspect("equal", adjustable="datalim")

    # Automatically adjust the axis limits to fit the data
    ax.relim()
    ax.autoscale_view()


    if output_filename is None:
        output_path = os.path.join(cfg.OUTPUTS_DIR, "plot.png")
    else:
        if not os.path.isabs(output_filename):
            output_path = os.path.join(cfg.OUTPUTS_DIR, output_filename)

    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"plot saved to {output_path}")

    if show:
        plt.show()


