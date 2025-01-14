import os
import h5py
import numpy as np
from matplotlib import pyplot as plt
import tbsim.utils.geometry_utils as GeoUtils


def load_scene_data(scene_key: str, data_path: str) -> dict:
    """
        Loads scene data from an hdf5 file.
        
        Args:
            scene_key: Key identifying the scene in the HDF5 file.
            data_path: Path to the HDF5 file.

        Returns:
            scene_data: A dictionary of dataset names and their corresponding data arrays for one scene.
                {
                    action_positions
                    action_sample_positions 
                    action_sample_yaws 
                    action_traj_positions 
                    action_traj_yaws 
                    action_yaws 
                    centroid 
                    extent 
                    scene_index 
                    track_id 
                    world_from_agent
                    yaw
                }
    """
    scene_data = {}
    
    with h5py.File(data_path, 'r') as file:
        if scene_key in file:
            for dataset_name in file[scene_key]:
                data = np.array(file[scene_key][dataset_name])
                scene_data[dataset_name] = data
        else:
            print(f"No data found for scene key: {scene_key}")

    return scene_data


def convert_to_world_coordinates(scene_data:  dict) -> np.ndarray:
    """
    Converts local (x, y) positions to world coordinates using a 2D transformation matrix.

    scene_data["action_traj_positions"][taj_index, time_frame, frame_index, [x, y]] -> world_positions[taj_index, time_frame, frame_index, [x, y]]

    Args:
        scene_data: A dictionary of dataset names and their corresponding data arrays for one scene.
            
    Returns:
        world_positions: Transformed world coordinates with the same shape as the input positions.
    """
    action_traj_positions = scene_data["action_traj_positions"]
    world_from_agent = scene_data["world_from_agent"]  

    world_positions = GeoUtils.batch_nd_transform_points_np(action_traj_positions, world_from_agent)
    return world_positions


def plot_world_trajectories(world_positions: np.ndarray,
                            ax: plt.Axes = None,
                            show: bool = True,
                            labels: list = None,
                            save_path: str = None) -> None:
    """
    Converts world positions into a concatenated trajectory and plots it on a Matplotlib axis.

    Args:
        world_positions: Array of world positions, where each element represents a trajectory.
        ax: Matplotlib axis to draw the trajectory. Creates a new axis if None.
        show: Whether to display the plot using plt.show().
        labels: Labels for the trajectories, if multiple. Must match the number of trajectories.
        save_path: Path to save the plot image. Defaults to 'plot.png' in the current working directory.

    Returns:
        None
    """
    # Convert world_positions to a single concatenated trajectory
    traj = np.concatenate([trajectory[0] for trajectory in world_positions], axis=0)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Plot the concatenated trajectory
    ax.plot(traj[:, 0], traj[:, 1], marker='o', label="Concatenated Trajectory")

    ax.set_aspect('equal', 'box')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("2D Trajectory Plot")
    ax.legend()

    # Determine file path in current working directory (if none provided)
    if save_path is None:
        save_path = os.path.join(os.getcwd(), "plot.png")
    else:
        # If user just gives a filename, place it in the CWD.
        if not os.path.isabs(save_path):
            save_path = os.path.join(os.getcwd(), save_path)

    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Plot saved to {save_path}")

    if show:
        plt.show()


def convert_data():
    """
        Using HumanML3D implementation, convert the data to HumanML3D vector representation.
        In particular: [num_of_traj, angular_velocity, linear_velocity_x, linear_velocity_z, y]
    """
    pass


def set_root_mask():
    """
        set HML_ROOT_HORIZONTAL_MASK as follow:
            [formatted data from trace,
              zero * for 260 dimensions]
    """
    pass


def generate_animation():
    """
        Using priorMDM implementation, generate animation for each of the trajectory.
    """
    pass


def main():
    scene_key = "scene_000010_orca_maps_31"
    data_path = "data.hdf5"

    scene_data = load_scene_data(scene_key=scene_key, data_path=data_path)
    world_positions = convert_to_world_coordinates(scene_data=scene_data)
    

    plot_world_trajectories(world_positions, show=False, labels=["Single agent"], save_path="my_trajectory_plot.png")
 

if __name__ == "__main__":
    main()  