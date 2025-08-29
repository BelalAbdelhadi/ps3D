from netCDF4 import Dataset
import numpy as np
from typing import List, Tuple, Union
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.patches as patches
import matplotlib.animation as animation

# Dictionary mapping energy levels to their associated file chunks
ENERGY_LEVEL_FILES = {
    "wp50": ['wp50-t1.nc', 'wp50-t2.nc', 'wp50-t3.nc'],
    "wp60": ['wp60-t1.nc', 'wp60-t2.nc', 'wp60-t3.nc'],
    "wp75": ['wp75-t1.nc', 'wp75-t2.nc', 'wp75-t3.nc'],
    "wp80": ['wp80-t1.nc', 'wp80-t2.nc', 'wp80-t3.nc'],
    "wp90": ['wp90-t1.nc', 'wp90-t2.nc', 'wp90-t3.nc']
}

def get_train_test_files(test_level: str) -> Tuple[List[str], List[str]]:
    """
    Given a test energy level (e.g., 'wp75'), return lists of training and test filenames.
    """
    train_files = []
    for level, files in ENERGY_LEVEL_FILES.items():
        if level != test_level:
            train_files.extend(files)
    test_files = ENERGY_LEVEL_FILES[test_level]
    return train_files, test_files


def load_nc_files(data_dir: str, train_files: List[str], test_files: List[str]) -> Tuple[List[Dataset], List[Dataset]]:
    """
    Load train and test netCDF files.
    Returns opened netCDF4.Dataset objects as lists.
    """
    nctrains = [Dataset(f"{data_dir}{f}", 'r') for f in train_files]
    nctest = [Dataset(f"{data_dir}{f}", 'r') for f in test_files]
    return nctrains, nctest


def load_variable_from_nc(data_dir: str, filename: str, var_name: str,
                          rec_slice=slice(0, 150),
                          y_slice=slice(0, 720),
                          x_slice=slice(0, 256)) -> np.ndarray:
    """
    Load and slice a variable from a netCDF file.
    """
    with Dataset(f"{data_dir}{filename}", 'r') as nc:
        data = np.squeeze(nc.variables[var_name][:])
        return data[rec_slice, y_slice, x_slice]


#Analysis functions

def create_animation_for_dataset(dataset, dataset_name, save_animation=True, z_level=0):
    """Create animation for a single dataset"""
    
    variables = ['u', 'v', 'w', 'p']
    
    # Get time levels from the dataset
    t_levels = None
    for var in variables:
        if var in dataset:
            t_levels = dataset[var].shape[0]
            break
    
    if t_levels is None:
        print(f"Error: No valid variables found in dataset {dataset_name}")
        return None, None
    
    # Compute color limits per variable across all levels and times for this dataset
    color_limits = {}
    for var in variables:
        if var in dataset:
            data_all = dataset[var][:t_levels, z_level, :, :]  # shape (t, y, x)
            color_limits[var] = (np.nanmin(data_all), np.nanmax(data_all))
        else:
            print(f"Warning: Variable {var} not found in dataset {dataset_name}")
            color_limits[var] = (0, 1)  # fallback
    
    # Create figure and subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axes = axs.ravel()
    images = []
    
    # Initialize plots
    for ax, var in zip(axes, variables):
        if var in dataset:
            data = dataset[var][0, z_level, :, :]
            vmin, vmax = color_limits[var]
            
            # Choose appropriate colormap based on variable
            if var in ['u', 'v']:  # velocity components
                cmap = 'RdBu_r'
            elif var == 'w':  # vertical velocity
                cmap = 'seismic'
            else:  # pressure or other
                cmap = 'viridis'
                
            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, 
                          aspect='auto', origin='lower')
            ax.set_title(f'{var} ({dataset_name})', fontsize=12, fontweight='bold')
            ax.set_xlabel('x (grid points)')
            ax.set_ylabel('y (grid points)')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
            
        else:
            # Handle missing variable
            im = ax.imshow(np.zeros((10, 10)), cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'{var} (NOT AVAILABLE)', fontsize=12, color='red')
            ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=20, color='red')
            
        images.append(im)
    
    fig.suptitle(f'Dataset: {dataset_name} - Time step: 0', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    def update(frame):
        """Update function for animation"""
        updated_artists = []
        for i, var in enumerate(variables):
            if var in dataset:
                data = dataset[var][frame, z_level, :, :]
                images[i].set_data(data)
                updated_artists.append(images[i])
        
        fig.suptitle(f'Dataset: {dataset_name} - Time step: {frame}', 
                    fontsize=16, fontweight='bold')
        return updated_artists
    
    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=t_levels, 
                                 interval=300, blit=False, repeat=True)
    
    if save_animation:
        # Create output directory
        output_dir = 'fig_idealized'
        os.makedirs(output_dir, exist_ok=True)
        """
        # Save as GIF (optional - requires pillow or imagemagick)
        try:
            gif_filename = os.path.join(output_dir, f'animation_{dataset_name}.gif')
            ani.save(gif_filename, writer='pillow', fps=3)
            print(f"Saved GIF animation: {gif_filename}")
        except Exception as e:
            print(f"Could not save GIF for {dataset_name}: {e}")
        """
        # Save as MP4 (optional - requires ffmpeg)
        try:
            mp4_filename = os.path.join(output_dir, f'animation_{dataset_name}_z={z_level}.mp4')
            ani.save(mp4_filename, writer='ffmpeg', fps=3, bitrate=1800)
            print(f"Saved MP4 animation: {mp4_filename}")
        except Exception as e:
            print(f"Could not save MP4 for {dataset_name}: {e}")
    
    return ani, fig
