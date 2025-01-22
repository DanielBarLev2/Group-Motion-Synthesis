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
    Converts local trajectories to world coordinates using a 2D transformation matrix.
    Then, add missing z-dimension to the world coordinates - set to zero.

    scene_data["action_traj_positions"][taj_index, time_frame_of_diffusion, time, [x, y]]
     ---> world_positions[taj_index, time, [x, y, z]]

    Args:
        scene_data: A dictionary of dataset names and their corresponding data arrays for one scene.
            
    Returns:
        world_positions: Transformed world coordinates.
    """
    action_traj_positions = scene_data["action_traj_positions"]
    world_from_agent = scene_data["world_from_agent"]  

    world_positions = GeoUtils.batch_nd_transform_points_np(action_traj_positions, world_from_agent)
    # select index 0 becuase it is the final trajectory; why - unknown?
    world_positions = world_positions[:, 0, :, :]

    z_dim = np.zeros((world_positions.shape[0], world_positions.shape[1], 1))
    world_positions = np.concatenate((world_positions, z_dim), axis=2)


    return world_positions


def plot_world_trajectories(trajectories: np.ndarray,
                            ax: plt.Axes = None,
                            show: bool = True,
                            save_path: str = None) -> None:
    """
    Plots 2D trajectories in world coordinates.
    Each trajectory is plotted with a different color.
    Args:
        trajectories: Array of world positions, where each element represents a trajectory.
        ax: Matplotlib axis to draw the trajectory. Creates a new axis if None.
        show: Whether to display the plot using plt.show().
        save_path: Path to save the plot image. Defaults to save in the current working directory.

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each trajectory in different colors
    for index in range(trajectories.shape[0]):
        ax.plot(trajectories[index, :, 0], trajectories[index, :, 1], marker='o', linestyle='-', 
                label=f"Trajectory {index+1}")

    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_title("2D Trajectories Plot")
    ax.legend(loc='best')

    if save_path is None:
        save_path = os.path.join(os.getcwd(), "plot.png")
    else:
        if not os.path.isabs(save_path):
            save_path = os.path.join(os.getcwd(), save_path)

    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Plot saved to {save_path}")

    if show:
        plt.show()


def convert_to_humanml3d_repr(positions, max_frames: int = 196):
    """
    Convert 3D Coordinates to HumanML3D representation.

    NOTE: HumanML3D representation refers to the ground plane as the x-z plane, not the x-y plane!
    x: horizontal axis (left-right), z: depth axis (front-back), y: vertical axis (up-down).
    Therefore, a transposition is required to convert the world coordinates to HumanML3D representation.

    world_positions[taj_index, time, [x, y, z]] 
    ---> root_data[taj_index, time, [angular_vel, x_vel, z_vel, y]]
    
    Args:
        positions: world coordinates.
        output_filename: Name of the output file to save the data.
        max_frames: Maximum number of frames to consider for each
               
    Returns:
        root_data: root data in HumanML3D representation.
    """
    # transpose the y and z coordinates
    positions = positions[:, :, [0, 2, 1]]

    batch_size, _, _ = positions.shape
    root_data = np.zeros((batch_size, max_frames, 4))

    for i in range(batch_size):
        pos = positions[i]  

        # v(t) = positions[t+1] - positions[t]
        velocity = pos[1:] - pos[:-1] 

        # Compute heading angle in the horizontal (x, z) plane
        heading = np.arctan2(pos[:, 0], pos[:, 2])

        # Angular velocity: difference in heading angles between consecutive frames
        r_velocity = heading[1:] - heading[:-1]

        l_velocity = velocity[:, [0, 2]]  
        
        root_y = pos[:, 1:2]

        # Combine into final format (N-1, 4)
        sequence_data = np.concatenate([r_velocity[:, None], l_velocity, root_y[:-1]], axis=-1)

        # Pad or cut the sequence to fit max_frames
        if sequence_data.shape[0] < max_frames:
            padding_rows = max_frames - sequence_data.shape[0]
            padding = np.zeros((padding_rows, 4))
            sequence_data = np.vstack([sequence_data, padding])
        else:
            sequence_data = sequence_data[:max_frames] 

        root_data[i] = sequence_data

    return root_data



def set_root_mask(root_data: list, output_filename="synthetic_data", batch_size=10, max_frames=196) -> np.ndarray:
    """
    Generate synthetic data by broadcasting root data across a batch and save it to a file.
    root_data[taj_index][time, [angular_vel, x_vel, z_vel, y]]
    ---> synthetic_data[batch_size, 263, 1, max_frames]

    NOTE:263 => 4 (root_data) + 259 (zeros)

    Args:
        root_data: Input data representing root positions or features. 
                   Expected to be of shape (4, max_frames) prior to reshaping.
        output_filename:Name of the output file to save the data.
        batch_size: Number of samples in the batch. Defaults to 10.
        max_frames: Maximum number of frames to consider for each sample. Defaults to 196.

    Saves the synthetic data to a npy file.
    """
    synthetic_data = np.zeros((batch_size, 263, 1, max_frames))

    # transpose from (4, max_frames) to (4, max_frames)
    root_data = np.transpose(root_data, (1, 0))
    # Add an additional axis for broadcasting (4, 1, max_frames)
    root_data = np.expand_dims(root_data, axis=2) 
    # Broadcast to match the batch size (batch_size, 4, 1, max_frames)
    root_data = np.stack([root_data] * batch_size, axis=0) 

    root_data = np.transpose(root_data, (0 , 1, 3, 2))
    synthetic_data[:, :4, :, :] = root_data  

    # Saves the output data
    output_path = os.path.join(os.getcwd(), output_filename)
    np.save(output_path, synthetic_data)
    print(f"Output data saved to: {output_path}")
    


def generate_animation():
    """
        Using priorMDM implementation, generate animation for each of the trajectory.
    """
    pass


def main():
    scene_key = "scene_000171_orca_maps_31"
    data_path = "data.hdf5"
    max_frames = 196

    scene_data = load_scene_data(scene_key=scene_key, data_path=data_path)

    world_positions = convert_to_world_coordinates(scene_data=scene_data)
    
    plot_world_trajectories(world_positions, show=False, save_path="trajectory_plot.png")

    humenml3d_data = convert_to_humanml3d_repr(positions=world_positions, max_frames=max_frames)

    set_root_mask(humenml3d_data[4])

if __name__ == "__main__":
    main()  