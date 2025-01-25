import os
import h5py
import numpy as np
from matplotlib import pyplot as plt
import tbsim.utils.geometry_utils as GeoUtils
from integration.src.generate_plots import plot_world_trajectories
from integration.config_files import cfg



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


def convert_to_world_coordinates(scene_data:  dict, output_filename: str = "init_positions") -> np.ndarray:
    """
    Converts local trajectories to world coordinates using a 2D transformation matrix.
    Then, add missing z-dimension to the world coordinates - set to zero.
    In addition, saves the initial positions to a file.

    scene_data["action_traj_positions"][taj_index, time_frame_of_diffusion, time, [x, y]]
     ---> world_positions[taj_index, time, [x, y, z]]

    Args:
        scene_data: A dictionary of dataset names and their corresponding data arrays for one scene.
        output_filename: Name of the output file to save the data.
    Returns:
        world_positions: Transformed world coordinates.
    """
    action_traj_positions = scene_data["action_traj_positions"]
    world_from_agent = scene_data["world_from_agent"]  

    world_positions = GeoUtils.batch_nd_transform_points_np(action_traj_positions, world_from_agent)
    # select index 0 becuase it is the final trajectory
    world_positions = world_positions[:, 0, :, :]

    z_dim = np.zeros((world_positions.shape[0], world_positions.shape[1], 1))
    world_positions = np.concatenate((world_positions, z_dim), axis=2)

    world_positions = world_positions * 10

    init_positions = world_positions[:, 0, :]
    init_positions = init_positions[:, [0, 2, 1]] 

    # Saves the output data
    output_path = os.path.join(cfg.SAVE_DIR, output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, init_positions)
    print(f"init_positions saved to: {output_path}")

    return world_positions



def convert_fps(world_positions: np.array, input_fps: int = 10, output_fps: int = 20) -> np.ndarray:
 
    """
    Converts trajectories from 10 FPS to 20 FPS by interpolating positions.
    
    Args:
        world_positions: world coordinates.
        input_fps (int): Original frames per second of the data.
        output_fps (int): Desired frames per second for the data.    
             
    Returns:
        interpolated_world_positions: New array with shape [traj_index, new_time, 3] at 20 FPS.
    """
    traj_count, time_steps, _ = world_positions.shape

    duration = time_steps / input_fps

    # generate original and new time steps
    original_times = np.linspace(0, duration, time_steps)
    new_time_steps = int(duration * output_fps)
    new_times = np.linspace(0, duration, new_time_steps)

    interpolated_world_positions = np.zeros((traj_count, new_time_steps, 3))

    for traj_index in range(traj_count):
        for coord in range(3):
            interpolated_world_positions[traj_index, :, coord] = np.interp(new_times, original_times, world_positions[traj_index, :, coord])

    return interpolated_world_positions


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

    batch_size, time, _ = positions.shape

    root_data = np.zeros((batch_size, max_frames, 4), dtype=np.float32)

    for i in range(batch_size):
        pos = positions[i]  

        # v(t) = positions[t+1] - positions[t]
        velocity = pos[1:] - pos[:-1] 

        # Compute heading angle in the horizontal (x, z) plane
        x_vel = velocity[:, 0]
        z_vel = velocity[:, 2]
        heading = np.arctan2(z_vel, x_vel)

        heading = np.unwrap(heading) # to avoid large jumps around ±pi

        # Angular velocity: difference in heading angles between consecutive frames
        r_velocity = np.zeros_like(heading)
        r_velocity[1:] = heading[1:] - heading[:-1]

        l_velocity = velocity[:, [0, 2]]  
        
        y_vals = pos[:-1, 1:2]

        # Combine into final format (N-1, 4)
        sequence_data = np.concatenate([r_velocity[:, None], x_vel[:, None], z_vel[:, None], y_vals], axis=-1)

        # Pad or cut the sequence to fit max_frames
        if sequence_data.shape[0] < max_frames:
            padding_rows = max_frames - sequence_data.shape[0]
            padding = np.zeros((padding_rows, 4))
            sequence_data = np.vstack([sequence_data, padding])
        else:
            sequence_data = sequence_data[:max_frames] 

        root_data[i] = sequence_data

    return root_data



def set_root_mask(root_data: list, output_filename="synthetic_data") -> np.ndarray:
    """
    Generate synthetic data by broadcasting root data across a batch and save it to a file.
    root_data[taj_index][time, [angular_vel, x_vel, z_vel, y]]
    ---> synthetic_data[batch_size, 263, 1, max_frames]

    NOTE:263 => 4 (root_data) + 259 (zeros)

    Args:
        root_data: Input data representing root positions or features. 
                   Expected to be of shape (4, max_frames) prior to reshaping.
        output_filename:Name of the output file to save the data.

    Saves the synthetic data to a npy file.
    """
    batch_size, time, _ = root_data.shape
    synthetic_data = np.zeros((batch_size, 263, 1, time))

    # transpose from (batch_size, 4, max_frames) to (batch_size, 4, max_frames) 
    root_data = np.transpose(root_data, (0, 2, 1))
    # שdd an additional axis for broadcasting (batch_size, 4, 1, max_frames)
    root_data = np.expand_dims(root_data, axis=2) 
    
    synthetic_data[:, :4, :, :] = root_data  

    # Saves the output data
    output_path = os.path.join(cfg.SAVE_DIR, output_filename)
    np.save(output_path, synthetic_data)
    print(f"synthetic data saved to: {output_path}")


def main():
    """
    Main function to process scene data and generate synthetic trajectory data.

    Steps:
    1. Load scene data from an HDF5 file using the provided scene key.
    2. Convert local trajectories to world coordinates and add a z-dimension.
    3. Adjust the FPS of the trajectories from 10 to 30 using interpolation.
    4. Plot and save the world trajectories as a 2D image.
    5. Convert world positions into the HumanML3D representation.
    6. Generate and save synthetic data in the expected format.
    """
    scene_data = load_scene_data(scene_key=cfg.SCENE_KEY, data_path=cfg.DATA_PATH)

    world_positions = convert_to_world_coordinates(scene_data=scene_data)

    world_positions = convert_fps(world_positions, input_fps=10, output_fps=30)
    
    plot_world_trajectories(world_positions, show=False, output_filename="trajectory_plot.png")

    humenml3d_data = convert_to_humanml3d_repr(positions=world_positions, max_frames=cfg.MAX_FRAMES)

    set_root_mask(root_data=humenml3d_data, output_filename="synthetic_data")


if __name__ == "__main__":
    main() 