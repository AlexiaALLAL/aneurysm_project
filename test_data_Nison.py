# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 10:25:40 2025

@author: 33650
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP threads
os.environ["MKL_NUM_THREADS"] = "1"  # MKL threads
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # numexpr threads
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # macOS Accelerate
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # OpenBLAS threads
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Prevent OpenMP duplication

import numpy as np
import meshio
import torch
import torch.nn as nn
from typing import List
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider



def xdmf_to_meshes(xdmf_file_path: str) -> List[meshio.Mesh]:
    """
    Opens an XDMF archive file, and extracts a data mesh object for every timestep.

    Args:
        xdmf_file_path (str): Path to the .xdmf file.

    Returns:
        List[meshio.Mesh]: List of data mesh objects for each timestep.
    """
    reader = meshio.xdmf.TimeSeriesReader(xdmf_file_path)
    points, cells = reader.read_points_cells()
    meshes = []

    # Extracting the meshes from the archive
    for i in range(reader.num_steps):
        try:
            time, point_data, cell_data, _ = reader.read_data(i)
        except ValueError:
            time, point_data, cell_data = reader.read_data(i)
        mesh = meshio.Mesh(points, cells, point_data=point_data, cell_data=cell_data)
        meshes.append(mesh)

    print(f"Loaded {len(meshes)} timesteps from {os.path.basename(xdmf_file_path)}\n")
    return meshes



def process_all_xdmf_files(data_dir: str, filename_pattern: str, file_range: range):
    """
    Processes multiple XDMF files to extract meshes for all timesteps.

    Args:
        data_dir (str): Directory containing the XDMF files.
        filename_pattern (str): Filename pattern (e.g., "AllFields_Resultats_MESH_{k}.xdmf").
        file_range (range): Range of file indices to process.

    Returns:
        List[meshio.Mesh]: List of meshes for all files and timesteps.
    """
    all_meshes = []

    for k in file_range:
        file_path = os.path.join(data_dir, filename_pattern.format(k=k))
        file_path=file_path.replace("\\","/")
        #print("FILEPATH",file_path)

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        print(f"Processing file: {file_path}")
        try:
            meshes = xdmf_to_meshes(file_path)
            all_meshes.extend(meshes)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print(f"Processed {len(all_meshes)} meshes from {len(file_range)} files.\n")
    return all_meshes




def normalize_and_interpolate(mesh, grid_resolution=64):
    """
    Normalizes and interpolates mesh data onto a uniform grid.

    Args:
        mesh (meshio.Mesh): A mesh object containing point and cell data.
        grid_resolution (int): Resolution of the uniform grid.

    Returns:
        dict: Dictionary containing interpolated velocity and pressure data.
    """
    # Extract spatial points and data
    points = mesh.points  # Shape: (n, 3)
    velocity = mesh.point_data.get("Vitesse")  # Shape: (n, d)
    pressure = mesh.point_data.get("Pression")  # Shape: (n,)

    if velocity is None or pressure is None:
        raise ValueError("Velocity or pressure data not found in the mesh.")

    # Normalize fields
    velocity_norm = (velocity - np.mean(velocity, axis=0)) / np.std(velocity, axis=0)
    pressure_norm = (pressure - np.mean(pressure)) / np.std(pressure)

    # Define uniform grid
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    z_min, z_max = points[:, 2].min(), points[:, 2].max()

    grid_x = np.linspace(x_min, x_max, grid_resolution)
    grid_y = np.linspace(y_min, y_max, grid_resolution)
    grid_z = np.linspace(z_min, z_max, grid_resolution)
    grid = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')  # 3D grid

    # Flatten the grid into a list of points for interpolation
    grid_points = np.vstack([g.ravel() for g in grid]).T  # Shape: (grid_resolution^3, 3)

    # Interpolate data onto the uniform grid
    velocity_grid = griddata(points, velocity_norm, grid_points, method="linear", fill_value=0)
    pressure_grid = griddata(points, pressure_norm, grid_points, method="linear", fill_value=0)

    # Reshape interpolated data back to the grid shape
    velocity_grid = velocity_grid.reshape(grid[0].shape + (velocity.shape[1],))  # Add channels for velocity components
    pressure_grid = pressure_grid.reshape(grid[0].shape)

    return {
        "velocity": velocity_grid,
        "pressure": pressure_grid,
    }


def grid_to_mesh(grid_velocity, grid_pressure, grid_resolution, bounds, cells):
    """
    Reconstructs a mesh from a uniform grid of velocity and pressure fields.

    Args:
        grid_velocity (np.ndarray): Interpolated velocity grid of shape [grid_res, grid_res, grid_res, 3].
        grid_pressure (np.ndarray): Interpolated pressure grid of shape [grid_res, grid_res, grid_res].
        grid_resolution (int): Resolution of the uniform grid.
        bounds (tuple): Min and max bounds of the original mesh (x_min, x_max, y_min, y_max, z_min, z_max).
        cells (list): Cell connectivity information from the original mesh.

    Returns:
        meshio.Mesh: Reconstructed mesh object.
    """
    x_min, x_max, y_min, y_max, z_min, z_max = bounds

    # Generate uniform grid points
    x = np.linspace(x_min, x_max, grid_resolution)
    y = np.linspace(y_min, y_max, grid_resolution)
    z = np.linspace(z_min, z_max, grid_resolution)
    uniform_grid_points = np.array(np.meshgrid(x, y, z, indexing="ij")).reshape(3, -1).T  # [grid_points, 3]

    # Flatten grid data for velocity and pressure
    velocity_flat = grid_velocity.reshape(-1, 3)  # [grid_points, 3]
    pressure_flat = grid_pressure.flatten()      # [grid_points]

    # Create the mesh
    mesh = meshio.Mesh(
        points=uniform_grid_points,  # [grid_points, 3]
        cells=cells,  # Reuse the original cell connectivity
        point_data={
            "velocity": velocity_flat,  # Velocity field
            "pressure": pressure_flat,  # Pressure field
        }
    )

    return mesh



def process_and_save(data_dir, filename_pattern, file_range, output_dir, grid_resolution=64):
    os.makedirs(output_dir, exist_ok=True)
    all_meshes = process_all_xdmf_files(data_dir, filename_pattern, file_range)
    for i, mesh in enumerate(all_meshes):
        try:
            data = normalize_and_interpolate(mesh, grid_resolution)
            np.save(os.path.join(output_dir, f"velocity_{i}.npy"), data["velocity"])
            np.save(os.path.join(output_dir, f"pressure_{i}.npy"), data["pressure"])
            print("done ",i,"/",len(all_meshes))
        except Exception as e:
            print(f"Error processing mesh {i}: {e}")
            #k=0

def process_and_save_time_series(data_dir, filename_pattern, file_range, output_dir, grid_resolution=64):
    os.makedirs(output_dir, exist_ok=True)
    
    for k in file_range:
        file_path = os.path.join(data_dir, filename_pattern.format(k=k)).replace("\\", "/")
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        print(f"Processing file: {file_path}")
        if os.path.isfile(os.path.join(output_dir, f"velocity_time_series_{k}.npy")) and os.path.isfile(os.path.join(output_dir, f"pressure_time_series_{k}.npy")):
            continue
        
        try:
            meshes = xdmf_to_meshes(file_path)  # Get all time steps for this file
            velocity_list, pressure_list = [], []

            for mesh in meshes:
                data = normalize_and_interpolate(mesh, grid_resolution)
                velocity_list.append(data["velocity"])
                pressure_list.append(data["pressure"])

            # Save time-series data as single files
            np.save(os.path.join(output_dir, f"velocity_time_series_{k}.npy"), np.stack(velocity_list))
            np.save(os.path.join(output_dir, f"pressure_time_series_{k}.npy"), np.stack(pressure_list))

            print(f"Saved time-series data for file {k}.")
        except Exception as e:
            print(f"Error processing file {k}: {e}")


# class BloodFlowDataset(torch.utils.data.Dataset):
#     def __init__(self, data_dir):
#         self.data_dir = data_dir
#         self.velocity_files = sorted(f for f in os.listdir(data_dir) if f.startswith("velocity"))
#         self.pressure_files = sorted(f for f in os.listdir(data_dir) if f.startswith("pressure"))

#     def __len__(self):
#         return len(self.velocity_files)

#     def __getitem__(self, idx):
#         velocity = np.load(os.path.join(self.data_dir, self.velocity_files[idx]))
#         pressure = np.load(os.path.join(self.data_dir, self.pressure_files[idx]))

#         velocity = torch.tensor(velocity, dtype=torch.float32)
#         pressure = torch.tensor(pressure, dtype=torch.float32)

#         return velocity, pressure

class BloodFlowDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.velocity_files = sorted(f for f in os.listdir(data_dir) if f.startswith("velocity_time_series"))
        self.pressure_files = sorted(f for f in os.listdir(data_dir) if f.startswith("pressure_time_series"))

    def __len__(self):
        return len(self.velocity_files)

    def __getitem__(self, idx):
        velocity = np.load(os.path.join(self.data_dir, self.velocity_files[idx]))
        pressure = np.load(os.path.join(self.data_dir, self.pressure_files[idx]))

        velocity = torch.tensor(velocity, dtype=torch.float32)  # Shape: (time_steps, x_dim, y_dim, z_dim, channels)
        pressure = torch.tensor(pressure, dtype=torch.float32)  # Shape: (time_steps, x_dim, y_dim, z_dim)

        return velocity, pressure


# Directory paths
DATA_DIR = "D:/Documents/cours/3a/IDSC/Data_challenge2/4Students_AnXplore03"
OUTPUT_DIR = "D:/Documents/cours/3a/IDSC/Data_challenge2/processed/data"

# Process and save data
FILENAME_PATTERN = "AllFields_Resultats_MESH_{k}.xdmf"
FILE_RANGE = range(1, 100)
GRID_RESOLUTION = 64

# process_and_save(DATA_DIR, FILENAME_PATTERN, FILE_RANGE, OUTPUT_DIR, GRID_RESOLUTION)
process_and_save_time_series(DATA_DIR, FILENAME_PATTERN, FILE_RANGE, OUTPUT_DIR, GRID_RESOLUTION)


# Load processed data
dataset = BloodFlowDataset(OUTPUT_DIR)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

####

def extract_data_from_dataloader(dataloader, num_batches=1, max_samples_per_batch=None):
    """
    Extracts velocity and pressure data from the DataLoader without flattening arrays.
    
    Args:
        dataloader: PyTorch DataLoader object.
        num_batches: Number of batches to extract for analysis.
        max_samples_per_batch: Maximum number of samples per batch to keep.

    Returns:
        velocity_array, pressure_array: Numpy arrays of concatenated velocity and pressure data.
    """
    velocity_list, pressure_list = [], []

    # Extract data from DataLoader
    for i, (velocity, pressure) in enumerate(dataloader):
        velocity_np = velocity.numpy()
        pressure_np = pressure.numpy()

        # Clean data (remove NaNs and infinities)
        velocity_np = velocity_np[np.isfinite(velocity_np)]
        pressure_np = pressure_np[np.isfinite(pressure_np)]

        # Reshape back to original dimensions after cleaning
        velocity_np = velocity_np.reshape(velocity.shape)
        pressure_np = pressure_np.reshape(pressure.shape)

        # Sample a subset of data for each batch if max_samples_per_batch is specified
        if max_samples_per_batch:
            batch_size = velocity_np.shape[0]
            sample_indices = np.random.choice(batch_size, min(max_samples_per_batch, batch_size), replace=False)
            velocity_np = velocity_np[sample_indices]
            pressure_np = pressure_np[sample_indices]

        velocity_list.append(velocity_np)
        pressure_list.append(pressure_np)

        if i + 1 == num_batches:
            break

    # Concatenate data along the batch dimension
    velocity_array = np.concatenate(velocity_list, axis=0)
    pressure_array = np.concatenate(pressure_list, axis=0)

    return velocity_array, pressure_array


# Extract sampled and cleaned data
velocity_sample, pressure_sample = extract_data_from_dataloader(dataloader, num_batches=5, max_samples_per_batch=200)



def plot_distributions_seaborn(velocity_data, pressure_data):
    """
    Plots distributions of velocity components (x, y, z) and pressure data using seaborn.

    Args:
        velocity_data: Numpy array containing velocity data in shape (batch_size, x_dim, y_dim, z_dim, 3).
        pressure_data: Numpy array containing pressure data in shape (batch_size, x_dim, y_dim, z_dim).
    """
    # Extract the velocity components
    velocity_x = velocity_data[..., 0]  # X-component of velocity
    velocity_y = velocity_data[..., 1]  # Y-component of velocity
    velocity_z = velocity_data[..., 2]  # Z-component of velocity

    # Flatten each component for plotting
    velocity_x_flat = velocity_x.flatten()
    velocity_y_flat = velocity_y.flatten()
    velocity_z_flat = velocity_z.flatten()
    pressure_flat = pressure_data.flatten()

    # Plot velocity X-component distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(velocity_x_flat, bins=50, kde=True, color="blue", label="Velocity X")
    plt.title("Velocity X-Component Distribution")
    plt.xlabel("Velocity X Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    # Plot velocity Y-component distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(velocity_y_flat, bins=50, kde=True, color="green", label="Velocity Y")
    plt.title("Velocity Y-Component Distribution")
    plt.xlabel("Velocity Y Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    # Plot velocity Z-component distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(velocity_z_flat, bins=50, kde=True, color="red", label="Velocity Z")
    plt.title("Velocity Z-Component Distribution")
    plt.xlabel("Velocity Z Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    # Plot pressure distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(pressure_flat, bins=50, kde=True, color="orange", label="Pressure")
    plt.title("Pressure Distribution")
    plt.xlabel("Pressure Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

# Call the seaborn plotting function
# plot_distributions_seaborn(velocity_sample, pressure_sample)


# def visualize_data_slices(velocity_data, pressure_data, slice_idx=0):
#     """
#     Visualizes slices of velocity and pressure fields.

#     Args:
#         velocity_data: Numpy array containing velocity data (not flattened).
#         pressure_data: Numpy array containing pressure data (not flattened).
#         slice_idx: Index of the slice to visualize.
#     """
#     # Validate slice index
#     if slice_idx >= velocity_data.shape[0]:
#         raise ValueError(f"Slice index {slice_idx} is out of range for velocity data.")

#     # Select the slice to visualize
#     velocity_slice = velocity_data[slice_idx, :, :, :]
#     pressure_slice = pressure_data[slice_idx, :, :, :]

#     # Compute the magnitude of velocity (if it's a vector field)
#     velocity_magnitude = np.linalg.norm(velocity_slice, axis=-1)  # Compute magnitude along the last dimension

#     # Visualize the middle slice along the z-axis
#     mid_z = velocity_magnitude.shape[2] // 2

#     # Plot velocity magnitude
#     plt.figure(figsize=(10, 5))
#     plt.imshow(velocity_magnitude[:, :, mid_z], origin="lower", cmap="viridis")
#     plt.colorbar(label="Velocity Magnitude")
#     plt.title(f"Velocity Magnitude Slice (Z-Mid: {mid_z}) for Sample {slice_idx}")
#     plt.xlabel("X")
#     plt.ylabel("Y")
#     plt.show()

#     # Plot pressure
#     plt.figure(figsize=(10, 5))
#     plt.imshow(pressure_slice[:, :, mid_z], origin="lower", cmap="coolwarm")
#     plt.colorbar(label="Pressure")
#     plt.title(f"Pressure Slice (Z-Mid: {mid_z}) for Sample {slice_idx}")
#     plt.xlabel("X")
#     plt.ylabel("Y")
#     plt.show()


# # Visualize slices (ensure `velocity_data` and `pressure_data` are full arrays, not samples)
# visualize_data_slices(velocity_sample, pressure_sample, slice_idx=0)

def visualize_time_series_slices(velocity_data, pressure_data, time_idx=0, slice_idx=0):
    """
    Visualizes slices of velocity and pressure fields for a specific time step.

    Args:
        velocity_data: Tensor of shape (time_steps, x_dim, y_dim, z_dim, channels).
        pressure_data: Tensor of shape (time_steps, x_dim, y_dim, z_dim).
        time_idx: Time step to visualize.
        slice_idx: Index of the slice to visualize.
    """
    if time_idx >= velocity_data.shape[0]:
        raise ValueError(f"Time index {time_idx} is out of range for the time series.")

    # Select the time step
    velocity_step = velocity_data[time_idx]
    pressure_step = pressure_data[time_idx]

    # Select the slice to visualize
    velocity_slice = velocity_step[:, :, :, :]
    pressure_slice = pressure_step[:, :, :]

    # Compute velocity magnitude
    velocity_magnitude = np.linalg.norm(velocity_slice, axis=-1)

    # Visualize the middle slice along the z-axis
    mid_z = velocity_magnitude.shape[2] // 2

    # Plot velocity magnitude
    plt.figure(figsize=(10, 5))
    plt.imshow(velocity_magnitude[:, :, mid_z], origin="lower", cmap="viridis")
    plt.colorbar(label="Velocity Magnitude")
    plt.title(f"Velocity Magnitude (Time Step: {time_idx}, Z-Slice: {mid_z})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

    # Plot pressure
    plt.figure(figsize=(10, 5))
    plt.imshow(pressure_slice[:, :, mid_z], origin="lower", cmap="coolwarm")
    plt.colorbar(label="Pressure")
    plt.title(f"Pressure (Time Step: {time_idx}, Z-Slice: {mid_z})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

#visualize_time_series_slices(velocity_sample[0], pressure_sample[0],time_idx=0, slice_idx=0)


def animate_time_series(velocity_data, pressure_data, slice_axis=2, interval=200):
    """
    Animates slices of velocity magnitude and pressure fields across all time steps.

    Args:
        velocity_data: Numpy array of shape (time_steps, x_dim, y_dim, z_dim, channels).
        pressure_data: Numpy array of shape (time_steps, x_dim, y_dim, z_dim).
        slice_axis: Axis to slice along (default: 2 for z-axis).
        interval: Time interval between frames in milliseconds (default: 200 ms).
    """
    # Get dimensions
    time_steps = velocity_data.shape[0]
    x_dim, y_dim, z_dim = velocity_data.shape[1:4]

    # Define the middle slice along the specified axis
    if slice_axis == 0:
        mid_slice = x_dim // 2
    elif slice_axis == 1:
        mid_slice = y_dim // 2
    elif slice_axis == 2:
        mid_slice = z_dim // 2
    else:
        raise ValueError(f"Invalid slice_axis: {slice_axis}. Must be 0, 1, or 2.")

    # Compute velocity magnitudes
    velocity_magnitude = np.linalg.norm(velocity_data, axis=-1)  # Shape: (time_steps, x_dim, y_dim, z_dim)

    # Create a figure with two subplots for velocity and pressure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Initialize plots
    if slice_axis == 0:
        velocity_slice = velocity_magnitude[0, mid_slice, :, :]
        pressure_slice = pressure_data[0, mid_slice, :, :]
    elif slice_axis == 1:
        velocity_slice = velocity_magnitude[0, :, mid_slice, :]
        pressure_slice = pressure_data[0, :, mid_slice, :]
    else:  # slice_axis == 2
        velocity_slice = velocity_magnitude[0, :, :, mid_slice]
        pressure_slice = pressure_data[0, :, :, mid_slice]

    velocity_im = ax1.imshow(velocity_slice, origin="lower", cmap="viridis")
    pressure_im = ax2.imshow(pressure_slice, origin="lower", cmap="coolwarm")

    ax1.set_title("Velocity Magnitude")
    ax2.set_title("Pressure")

    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")

    plt.colorbar(velocity_im, ax=ax1, label="Velocity Magnitude")
    plt.colorbar(pressure_im, ax=ax2, label="Pressure")

    # Update function for animation
    def update(frame):
        if slice_axis == 0:
            velocity_slice = velocity_magnitude[frame, mid_slice, :, :]
            pressure_slice = pressure_data[frame, mid_slice, :, :]
        elif slice_axis == 1:
            velocity_slice = velocity_magnitude[frame, :, mid_slice, :]
            pressure_slice = pressure_data[frame, :, mid_slice, :]
        else:  # slice_axis == 2
            velocity_slice = velocity_magnitude[frame, :, :, mid_slice]
            pressure_slice = pressure_data[frame, :, :, mid_slice]

        velocity_im.set_array(velocity_slice)
        pressure_im.set_array(pressure_slice)
        ax1.set_title(f"Velocity Magnitude (Time Step: {frame})")
        ax2.set_title(f"Pressure (Time Step: {frame})")
        return velocity_im, pressure_im

    # Create animation
    anim = FuncAnimation(fig, update, frames=time_steps, interval=interval, blit=False)

    plt.tight_layout()
    plt.show()
    anim.save('D:/Documents/cours/3a/IDSC/Data_challenge2/processed/time_series_animation.gif', fps=5)

    
animate_time_series(velocity_sample[0], pressure_sample[0], slice_axis=2, interval=200)

def interactive_time_series(velocity_data, pressure_data, slice_axis=2,num_slice=None):
    """
    Interactive visualization of velocity magnitude and pressure fields with a time slider.

    Args:
        velocity_data: Numpy array of shape (time_steps, x_dim, y_dim, z_dim, channels).
        pressure_data: Numpy array of shape (time_steps, x_dim, y_dim, z_dim).
        slice_axis: Axis to slice along (default: 2 for z-axis).
    """
    
    plt.ion()

    # Get dimensions
    time_steps = velocity_data.shape[0]
    x_dim, y_dim, z_dim = velocity_data.shape[1:4]

    # Define the middle slice along the specified axis
    if num_slice==None:
        if slice_axis == 0:
            mid_slice = x_dim // 2
        elif slice_axis == 1:
            mid_slice = y_dim // 2
        elif slice_axis == 2:
            mid_slice = z_dim // 2
        else:
            raise ValueError(f"Invalid slice_axis: {slice_axis}. Must be 0, 1, or 2.")
    else:
        mid_slice=num_slice
    # Compute velocity magnitudes
    velocity_magnitude = np.linalg.norm(velocity_data, axis=-1)  # Shape: (time_steps, x_dim, y_dim, z_dim)

    # Create a figure with two subplots for velocity and pressure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Initialize plots
    if slice_axis == 0:
        velocity_slice = velocity_magnitude[0, mid_slice, :, :]
        pressure_slice = pressure_data[0, mid_slice, :, :]
    elif slice_axis == 1:
        velocity_slice = velocity_magnitude[0, :, mid_slice, :]
        pressure_slice = pressure_data[0, :, mid_slice, :]
    else:  # slice_axis == 2
        velocity_slice = velocity_magnitude[0, :, :, mid_slice]
        pressure_slice = pressure_data[0, :, :, mid_slice]

    velocity_im = ax1.imshow(velocity_slice, origin="lower", cmap="viridis")
    pressure_im = ax2.imshow(pressure_slice, origin="lower", cmap="coolwarm")

    ax1.set_title("Velocity Magnitude (Time Step: 0)")
    ax2.set_title("Pressure (Time Step: 0)")
    if slice_axis==2:
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
    elif slice_axis==1:
        ax1.set_xlabel("X")
        ax1.set_ylabel("Z")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Z")
    elif slice_axis==0:
        ax1.set_xlabel("Y")
        ax1.set_ylabel("Z")
        ax2.set_xlabel("Y")
        ax2.set_ylabel("Z")
    

    plt.colorbar(velocity_im, ax=ax1, label="Velocity Magnitude")
    plt.colorbar(pressure_im, ax=ax2, label="Pressure")

    # Add slider for time selection
    slider_ax = plt.axes([0.25, 0.02, 0.5, 0.03])  # x, y, width, height
    time_slider = Slider(slider_ax, 'Time Step', 0, time_steps - 1, valinit=0, valstep=1)

    # Update function for the slider
    def update(frame):
        frame = int(frame)  # Ensure frame is an integer
        if slice_axis == 0:
            velocity_slice = velocity_magnitude[frame, mid_slice, :, :]
            pressure_slice = pressure_data[frame, mid_slice, :, :]
        elif slice_axis == 1:
            velocity_slice = velocity_magnitude[frame, :, mid_slice, :]
            pressure_slice = pressure_data[frame, :, mid_slice, :]
        else:  # slice_axis == 2
            velocity_slice = velocity_magnitude[frame, :, :, mid_slice]
            pressure_slice = pressure_data[frame, :, :, mid_slice]

        velocity_im.set_array(velocity_slice)
        pressure_im.set_array(pressure_slice)
        ax1.set_title(f"Velocity Magnitude (Time Step: {frame})")
        ax2.set_title(f"Pressure (Time Step: {frame})")
        fig.canvas.draw_idle()

    # Connect the slider to the update function
    time_slider.on_changed(update)

    plt.tight_layout()
    plt.show()
    return time_slider
    
#keep_slider=interactive_time_series(velocity_sample[0], pressure_sample[0], slice_axis=0,num_slice=32)

def interactive_time_series2(velocity_data, pressure_data, slice_axis=2, num_slice=None):
    """
    Interactive visualization of velocity magnitude and pressure fields with time and slice sliders.

    Args:
        velocity_data: Numpy array of shape (time_steps, x_dim, y_dim, z_dim, channels).
        pressure_data: Numpy array of shape (time_steps, x_dim, y_dim, z_dim).
        slice_axis: Axis to slice along (default: 2 for z-axis).
        num_slice: Initial slice index (if None, uses the middle slice).
    """

    plt.ion()

    # Get dimensions
    time_steps = velocity_data.shape[0]
    x_dim, y_dim, z_dim = velocity_data.shape[1:4]
    axis_dim = [x_dim, y_dim, z_dim][slice_axis]  # Dimension of the selected axis

    # Set the initial slice
    mid_slice = axis_dim // 2 if num_slice is None else num_slice

    # Compute velocity magnitudes
    velocity_magnitude = np.linalg.norm(velocity_data, axis=-1)  # Shape: (time_steps, x_dim, y_dim, z_dim)

    # Create a figure with two subplots for velocity and pressure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Initialize plots
    def get_slices(frame, slice_idx):
        if slice_axis == 0:
            return (
                velocity_magnitude[frame, slice_idx, :, :],
                pressure_data[frame, slice_idx, :, :],
            )
        elif slice_axis == 1:
            return (
                velocity_magnitude[frame, :, slice_idx, :],
                pressure_data[frame, :, slice_idx, :],
            )
        else:  # slice_axis == 2
            return (
                velocity_magnitude[frame, :, :, slice_idx],
                pressure_data[frame, :, :, slice_idx],
            )

    velocity_slice, pressure_slice = get_slices(0, mid_slice)

    velocity_im = ax1.imshow(velocity_slice, origin="lower", cmap="viridis")
    pressure_im = ax2.imshow(pressure_slice, origin="lower", cmap="coolwarm")

    ax1.set_title("Velocity Magnitude (Time Step: 0)")
    ax2.set_title("Pressure (Time Step: 0)")

    # Set axis labels based on slice_axis
    if slice_axis == 2:
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
    elif slice_axis == 1:
        ax1.set_xlabel("X")
        ax1.set_ylabel("Z")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Z")
    elif slice_axis == 0:
        ax1.set_xlabel("Y")
        ax1.set_ylabel("Z")
        ax2.set_xlabel("Y")
        ax2.set_ylabel("Z")

    plt.colorbar(velocity_im, ax=ax1, label="Velocity Magnitude")
    plt.colorbar(pressure_im, ax=ax2, label="Pressure")

    # Add sliders
    time_slider_ax = plt.axes([0.25, 0.02, 0.5, 0.03])  # x, y, width, height
    slice_slider_ax = plt.axes([0.25, 0.06, 0.5, 0.03])  # x, y, width, height

    time_slider = Slider(time_slider_ax, "Time Step", 0, time_steps - 1, valinit=0, valstep=1)
    slice_slider = Slider(slice_slider_ax, "Slice Index", 0, axis_dim - 1, valinit=mid_slice, valstep=1)

    # Update function for sliders
    def update(_):
        frame = int(time_slider.val)
        slice_idx = int(slice_slider.val)

        velocity_slice, pressure_slice = get_slices(frame, slice_idx)

        velocity_im.set_array(velocity_slice)
        pressure_im.set_array(pressure_slice)

        ax1.set_title(f"Velocity Magnitude (Time Step: {frame}, Slice: {slice_idx})")
        ax2.set_title(f"Pressure (Time Step: {frame}, Slice: {slice_idx})")
        fig.canvas.draw_idle()

    # Connect sliders to the update function
    time_slider.on_changed(update)
    slice_slider.on_changed(update)

    plt.tight_layout()
    plt.show()

    return time_slider, slice_slider

time_slider,slice_slider= interactive_time_series2(velocity_sample[0], pressure_sample[0], slice_axis=2)

class ConvLSTMModel(nn.Module):
    def __init__(self, in_channels, hidden_size, kernel_size=3, num_layers=1):
        super(ConvLSTMModel, self).__init__()
        self.conv = nn.Conv3d(in_channels, hidden_size, kernel_size=kernel_size, padding=1)
        self.pool1 = nn.AvgPool3d(kernel_size=2, stride=2)
        self.pool2 = nn.AvgPool3d(kernel_size=2, stride=2)

        self.hidden_size = hidden_size
        self.lstm = None
        self.fc_reconstruct = None

        # Decoders for velocity and pressure
        self.upconv1_velocity = nn.ConvTranspose3d(hidden_size, hidden_size, kernel_size=2, stride=2)
        self.upconv2_velocity = nn.ConvTranspose3d(hidden_size, 3, kernel_size=2, stride=2)  # Velocity channels
        self.upconv1_pressure = nn.ConvTranspose3d(hidden_size, hidden_size, kernel_size=2, stride=2)
        self.upconv2_pressure = nn.ConvTranspose3d(hidden_size, 1, kernel_size=2, stride=2)  # Single pressure channel

    def forward(self, x):
        batch, seq_len, channels, x_dim, y_dim, z_dim = x.shape

        # Process each frame through convolutional and pooling layers
        conv_out = []
        for t in range(seq_len):
            x_t = x[:, t]  # [batch, channels, x_dim, y_dim, z_dim]
            x_t = self.conv(x_t)
            x_t = self.pool1(x_t)
            x_t = self.pool2(x_t)
            conv_out.append(x_t.flatten(1))  # Flatten spatial dimensions

        # Stack for LSTM input
        lstm_input = torch.stack(conv_out, dim=1)  # [batch, seq_len, reduced_feature_size]

        # Dynamically initialize LSTM and reconstruction layer
        if self.lstm is None:
            reduced_feature_size = lstm_input.shape[-1]
            self.lstm = nn.LSTM(reduced_feature_size, self.hidden_size, batch_first=True)
            self.fc_reconstruct = nn.Linear(self.hidden_size, reduced_feature_size)

        # LSTM forward pass
        lstm_out, _ = self.lstm(lstm_input)
        lstm_out = lstm_out[:, -1, :]  # Use only the last time step

        # Reconstruct spatial dimensions
        lstm_out_reconstructed = self.fc_reconstruct(lstm_out)
        lstm_out_reconstructed = lstm_out_reconstructed.view(batch, self.hidden_size, x_dim // 4, y_dim // 4, z_dim // 4)

        # Decode velocity
        velocity_decoded = self.upconv1_velocity(lstm_out_reconstructed)
        velocity_pred = self.upconv2_velocity(velocity_decoded)

        # Decode pressure
        pressure_decoded = self.upconv1_pressure(lstm_out_reconstructed)
        pressure_pred = self.upconv2_pressure(pressure_decoded)

        return velocity_pred, pressure_pred


class BloodFlowDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, seq_len=2, target_offset=1):
        """
        Args:
            data_dir (str): Directory containing processed .npy files.
            seq_len (int): Number of frames in input time series (default: 2).
            target_offset (int): Frame offset for the target (default: 1, predicts the next frame).
        """
        self.data_dir = data_dir
        self.velocity_files = sorted(f for f in os.listdir(data_dir) if f.startswith("velocity_time_series"))
        self.pressure_files = sorted(f for f in os.listdir(data_dir) if f.startswith("pressure_time_series"))
        self.seq_len = seq_len
        self.target_offset = target_offset

    def __len__(self):
        return len(self.velocity_files) * (80 - self.seq_len - self.target_offset + 1)

    def __getitem__(self, idx):
        # Determine which file and frame range the index corresponds to
        file_idx = idx // (80 - self.seq_len - self.target_offset + 1)
        frame_start = idx % (80 - self.seq_len - self.target_offset + 1)

        # Load velocity and pressure data
        velocity = np.load(os.path.join(self.data_dir, self.velocity_files[file_idx]))  # Shape: (80, grid_res, grid_res, grid_res, channels)
        pressure = np.load(os.path.join(self.data_dir, self.pressure_files[file_idx]))  # Shape: (80, grid_res, grid_res, grid_res)

        # Extract input frames (seq_len) and the target frame
        input_velocity = velocity[frame_start : frame_start + self.seq_len]  # Shape: (seq_len, grid_res, grid_res, grid_res, channels)
        input_pressure = pressure[frame_start : frame_start + self.seq_len]  # Shape: (seq_len, grid_res, grid_res, grid_res)
        target_velocity = velocity[frame_start + self.seq_len + self.target_offset - 1]  # Shape: (grid_res, grid_res, grid_res, channels)
        target_pressure = pressure[frame_start + self.seq_len + self.target_offset - 1]  # Shape: (grid_res, grid_res, grid_res)

        return torch.tensor(input_velocity, dtype=torch.float32), \
               torch.tensor(input_pressure, dtype=torch.float32), \
               torch.tensor(target_velocity, dtype=torch.float32), \
               torch.tensor(target_pressure, dtype=torch.float32)


# Parameters
seq_len = 2  # Two input frames to predict the third
target_offset = 1
in_channels = 4  # Velocity (3) + Pressure (1)
hidden_size = 32
num_layers = 1
num_epochs = 2
batch_size = 2
learning_rate = 0.001

# Dataset and DataLoader
dataset = BloodFlowDataset(OUTPUT_DIR, seq_len=seq_len, target_offset=target_offset)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model, optimizer, and loss function
model = ConvLSTMModel(in_channels=in_channels, hidden_size=hidden_size, num_layers=num_layers)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# Train the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for input_velocity, input_pressure, target_velocity, target_pressure in dataloader:
        input_velocity = input_velocity.to(device)  # [batch, seq_len, x_dim, y_dim, z_dim, channels]
        input_pressure = input_pressure.to(device)  # [batch, seq_len, x_dim, y_dim, z_dim]
        target_velocity = target_velocity.to(device)  # [batch, x_dim, y_dim, z_dim, channels]
        target_pressure = target_pressure.to(device)  # [batch, x_dim, y_dim, z_dim]
    
        # Combine velocity and pressure inputs
        input_pressure = input_pressure.unsqueeze(-1)  # Add channel dim: [batch, seq_len, x_dim, y_dim, z_dim, 1]
        input_data = torch.cat([input_velocity, input_pressure], dim=-1)  # [batch, seq_len, x_dim, y_dim, z_dim, 4]
        input_data = input_data.permute(0, 1, 5, 2, 3, 4)  # Rearrange for Conv3D
    
        # Forward pass
        velocity_pred, pressure_pred = model(input_data)
    
        # Permute velocity_pred to match target_velocity shape
        velocity_pred = velocity_pred.permute(0, 2, 3, 4, 1)  # [batch, x_dim, y_dim, z_dim, channels]
    
        # Compute loss
        loss_velocity = loss_fn(velocity_pred, target_velocity)
        loss_pressure = loss_fn(pressure_pred, target_pressure)
        loss = loss_velocity + loss_pressure
    
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        total_loss += loss.item()



    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")


# Example usage
model.eval()
model.to(device)
# Load a processed time series
data_dir = "D:/Documents/cours/3a/IDSC/Data_challenge2/processed/data"
velocity_data = np.load(f"{data_dir}/velocity_time_series_1.npy")  # Shape: [80, grid_res, grid_res, grid_res, 3]
pressure_data = np.load(f"{data_dir}/pressure_time_series_1.npy")  # Shape: [80, grid_res, grid_res, grid_res]

# Select two consecutive frames for prediction
frame_1 = 0
frame_2 = 1
input_velocity = velocity_data[frame_1:frame_2 + 1]  # Shape: [2, grid_res, grid_res, grid_res, 3]
input_pressure = pressure_data[frame_1:frame_2 + 1]  # Shape: [2, grid_res, grid_res, grid_res]

# Convert to tensors
input_velocity = torch.tensor(input_velocity, dtype=torch.float32).unsqueeze(0).to(device)  # [1, 2, grid_res, grid_res, grid_res, 3]
input_pressure = torch.tensor(input_pressure, dtype=torch.float32).unsqueeze(0).to(device)  # [1, 2, grid_res, grid_res, grid_res]

# Add a channel dimension to pressure and concatenate inputs
input_pressure = input_pressure.unsqueeze(-1)  # [1, 2, grid_res, grid_res, grid_res, 1]
input_data = torch.cat([input_velocity, input_pressure], dim=-1)  # [1, 2, grid_res, grid_res, grid_res, 4]
input_data = input_data.permute(0, 1, 5, 2, 3, 4)  # [1, 2, 4, grid_res, grid_res, grid_res]

# Make predictions
with torch.no_grad():
    velocity_pred, pressure_pred = model(input_data)

# Adjust shapes for visualization or further processing
velocity_pred = velocity_pred.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()  # [grid_res, grid_res, grid_res, 3]
pressure_pred = pressure_pred.squeeze(0).squeeze(0).cpu().numpy()  # [grid_res, grid_res, grid_res]

print(f"Predicted Velocity Shape: {velocity_pred.shape}")
print(f"Predicted Pressure Shape: {pressure_pred.shape}")


def plot_frames(input_velocity, input_pressure, velocity_pred, pressure_pred, slice_axis=2):
    """
    Plot slices of input frames and the predicted frame for velocity and pressure.

    Args:
        input_velocity (torch.Tensor): Input velocity tensor of shape [1, 2, grid_res, grid_res, grid_res, 3].
        input_pressure (torch.Tensor): Input pressure tensor of shape [1, 2, grid_res, grid_res, grid_res].
        velocity_pred (np.ndarray): Predicted velocity tensor of shape [grid_res, grid_res, grid_res, 3].
        pressure_pred (np.ndarray): Predicted pressure tensor of shape [grid_res, grid_res, grid_res].
        slice_axis (int): Axis to slice along (default: 2 for the z-axis).
    """
    # Convert tensors to numpy arrays
    input_velocity = input_velocity.squeeze(0).cpu().numpy()  # Shape: [2, grid_res, grid_res, grid_res, 3]
    input_pressure = input_pressure.squeeze(0).cpu().numpy()  # Shape: [2, grid_res, grid_res, grid_res]

    # Define the middle slice along the specified axis
    mid_slice = velocity_pred.shape[slice_axis] // 2

    # Slice the input frames
    velocity_input_1 = input_velocity[0, :, :, mid_slice, 0]  # First frame, first velocity component
    velocity_input_2 = input_velocity[1, :, :, mid_slice, 0]  # Second frame, first velocity component
    velocity_output = velocity_pred[:, :, mid_slice, 0]  # Predicted frame, first velocity component

    pressure_input_1 = input_pressure[0, :, :, mid_slice]  # First frame pressure
    pressure_input_2 = input_pressure[1, :, :, mid_slice]  # Second frame pressure
    pressure_output = pressure_pred[:, :, mid_slice]  # Predicted pressure

    # Plot velocity slices
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(velocity_input_1, origin="lower", cmap="viridis")
    axs[0].set_title("Velocity Input 1")
    axs[1].imshow(velocity_input_2, origin="lower", cmap="viridis")
    axs[1].set_title("Velocity Input 2")
    axs[2].imshow(velocity_output, origin="lower", cmap="viridis")
    axs[2].set_title("Velocity Predicted")
    plt.colorbar(axs[2].images[0], ax=axs, orientation="vertical")
    plt.tight_layout()
    plt.show()

    # Plot pressure slices
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(pressure_input_1, origin="lower", cmap="coolwarm")
    axs[0].set_title("Pressure Input 1")
    axs[1].imshow(pressure_input_2, origin="lower", cmap="coolwarm")
    axs[1].set_title("Pressure Input 2")
    axs[2].imshow(pressure_output, origin="lower", cmap="coolwarm")
    axs[2].set_title("Pressure Predicted")
    plt.colorbar(axs[2].images[0], ax=axs, orientation="vertical")
    plt.tight_layout()
    plt.show()

plot_frames(input_velocity, input_pressure, velocity_pred, pressure_pred, slice_axis=2)
