import os
import h5py
import numpy as np
from matplotlib import pyplot as plt
import tbsim.utils.geometry_utils as GeoUtils


def load_scene_data(scene_key):
    """
    Reads scene data for a specified scene from an HDF5 file.
    
    @param: scene_key (str): Key for the scene to load data from.

    Returns:
        dict: A dictionary containing data arrays for all datasets within the specified scene.
    """
    file_path = "data.hdf5"
    scene_data = {}
    
    with h5py.File(file_path, 'r') as file:
           
        # Check if the scene key exists in the file
        if scene_key in file:
            # Iterate over each dataset within the scene group
            for dataset_name in file[scene_key]:
                
                data = np.array(file[scene_key][dataset_name])

                print(f'{dataset_name} shape of: {data.shape}')

                scene_data[dataset_name] = data
        else:
            print(f"No data found for scene key: {scene_key}")

    return scene_data


def convert_to_world_coordinates(scene_data):
    """
    Convert local (x,y) positions to world coordinates using a 2D transform in TBSim.
    """
    # shape: (1, 50, 52, 2)
    action_traj_positions = scene_data["action_traj_positions"]
    # shape: (1, 50, 3, 3), which is a 2D transform in homogeneous form
    world_from_agent = scene_data["world_from_agent"]  

    # Directly pass 2D points to TBSim's function (without adding z=0):
    # This way, batch_nd_transform_points_np recognizes a 2D transform.
    world_positions = GeoUtils.batch_nd_transform_points_np(
        action_traj_positions,   # shape (...,2)
        world_from_agent         # shape (...,3,3)
    )
    return world_positions


def plot_trajectories(
    trajectories_2d, 
    ax=None, 
    show=True, 
    labels=None,
    save_path=None,
):
    """
    Plot one or more 2D trajectories on a Matplotlib axis, then save to a file in the CWD.

    Parameters
    ----------
    trajectories_2d : np.ndarray
        Either shape (T, 2) for a single trajectory or shape (N, T, 2) for N trajectories.
    ax : matplotlib.axes.Axes, optional
        Axis on which to draw the trajectories. If None, a new figure + axis is created.
    show : bool, optional
        Whether to call plt.show() at the end of the function.
    labels : list of str, optional
        If multiple trajectories, labels for each agent/trajectory. Must have length N if provided.
    save_path : str, optional
        File name (or full path) to save the figure. If None, defaults to 'plot.png' in CWD.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Convert (T,2) -> (1,T,2) if input is a single trajectory
    if len(trajectories_2d.shape) == 2 and trajectories_2d.shape[-1] == 2:
        trajectories_2d = trajectories_2d[np.newaxis, ...]

    num_trajs = trajectories_2d.shape[0]
    for i in range(num_trajs):
        traj = trajectories_2d[i]  # shape (T,2)
        label_str = labels[i] if (labels is not None and i < len(labels)) else f"Traj {i}"
        ax.plot(traj[:, 0], traj[:, 1], marker='o', label=label_str)

    ax.set_aspect('equal', 'box')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("2D Trajectory Plot")
    if labels is not None:
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
    scene_key = "scene_000040_orca_maps_31"
    scene_data = load_scene_data(scene_key)
    world_positions = convert_to_world_coordinates(scene_data)

    print("World positions shape:", world_positions.shape)
   
    single_trajectory = world_positions[1, 0]

    plot_trajectories(single_trajectory, show=False, labels=["Single agent"], save_path="my_trajectory_plot.png")

    
   

    

if __name__ == "__main__":
    # scene_data["action_traj_positions"][taj_index, time_frame]
    main()  