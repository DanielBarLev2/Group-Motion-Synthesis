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

    # ###
    # world_positions = world_positions * 10
    # ###

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


def convert_to_humanml3d_repr(positions, output_filename: str = "output", max_frames: int = 196):
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

    # Save the output data
    output_path = os.path.join(os.getcwd(), output_filename)
    np.save(output_path, root_data)
    print(f"Output data saved to: {output_path}")

    return root_data


def convert_data(world_positions: np.ndarray, output_filename: str = "output.mpy", max_frame: int = 196) -> np.ndarray:
    """
    joint_num = 22
      def get_cont6d_params(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        
        recover_from_ric
        # .mpy
        Using HumanML3D implementation, convert the data to HumanML3D vector representation.
        In particular: [num_of_traj, angular_velocity, linear_velocity_x, linear_velocity_z, y]
    """
    


def set_root_mask(trace_data: list) -> np.ndarray:
    """
        set HML_ROOT_HORIZONTAL_MASK as follow:
            [formatted data from trace,
              zero * for 260 dimensions]
    """
    # trace_array = np.array(trace_data)
    # return np.concatenate((trace_array, np.zeros(260)))
    pass


def generate_animation():
    """
        Using priorMDM implementation, generate animation for each of the trajectory.
    """
    pass


def main():
    scene_key = "scene_000171_orca_maps_31"
    data_path = "data.hdf5"
    output_filename = "output"
    max_frames = 196

    scene_data = load_scene_data(scene_key=scene_key, data_path=data_path)

    world_positions = convert_to_world_coordinates(scene_data=scene_data)
    
    plot_world_trajectories(world_positions, show=False, save_path="trajectory_plot.png")

    humenml3d_data = convert_to_humanml3d_repr(world_positions=world_positions, output_filename=output_filename, max_frames=max_frames)


    # print("World positions shape:", world_positions.shape)
    # print(world_positions[4])
    # print("Output shape:", humenml3d_data.shape)
    # print("[angular_vel, x_vel, z_vel, y]:")
    # print(humenml3d_data[4])

    # num_frames = 196
    # movment = np.zeros((4, num_frames))
    # movment[0, :] = 2
    # movment[1, :] = 0
    # movment[2, :] = np.pi/2
    # movment[3, :] = 0

    # output_path = os.path.join(os.getcwd(), "output")
    # np.save(output_path, movment)
    # print(f"Output data saved to: {output_path}")

    # print(move_left_tensor)

if __name__ == "__main__":
    main()  