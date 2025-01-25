import os
import numpy as np
import matplotlib.pyplot as plt

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


