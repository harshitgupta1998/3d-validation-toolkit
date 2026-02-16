"""
Utility functions for the 3D validation toolkit.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Union, Tuple, Any

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


def load_point_cloud(file_path: str) -> Any:
    """
    Load point cloud from file.
    
    Args:
        file_path: Path to .ply, .pcd, or other supported format
        
    Returns:
        Open3D point cloud
    """
    if not HAS_OPEN3D:
        raise ImportError("Open3D is required. Install with: pip install open3d")
    return o3d.io.read_point_cloud(file_path)


def load_mesh(file_path: str) -> Any:
    """
    Load mesh from file.
    
    Args:
        file_path: Path to mesh file (.obj, .ply, .stl, etc.)
        
    Returns:
        Trimesh object
    """
    if not HAS_TRIMESH:
        raise ImportError("Trimesh is required. Install with: pip install trimesh")
    return trimesh.load(file_path)


def save_point_cloud(pcd: Any,
                     file_path: str):
    """
    Save point cloud to file.
    
    Args:
        pcd: Point cloud to save (open3d.geometry.PointCloud)
        file_path: Output file path
    """
    if not HAS_OPEN3D:
        raise ImportError("Open3D is required. Install with: pip install open3d")
    o3d.io.write_point_cloud(file_path, pcd)


def downsample_point_cloud(pcd: Any,
                          voxel_size: float) -> Any:
    """
    Downsample point cloud using voxel grid.
    
    Args:
        pcd: Input point cloud (open3d.geometry.PointCloud)
        voxel_size: Size of voxels
        
    Returns:
        Downsampled point cloud
    """
    if not HAS_OPEN3D:
        raise ImportError("Open3D is required. Install with: pip install open3d")
    return pcd.voxel_down_sample(voxel_size)


def normalize_point_cloud(pcd: Any) -> Any:
    """
    Normalize point cloud to unit cube.
    
    Args:
        pcd: Input point cloud (open3d.geometry.PointCloud)
        
    Returns:
        Normalized point cloud
    """
    if not HAS_OPEN3D:
        raise ImportError("Open3D is required. Install with: pip install open3d")
    
    points = np.asarray(pcd.points)
    
    # Center
    centroid = np.mean(points, axis=0)
    points -= centroid
    
    # Scale
    max_dist = np.max(np.linalg.norm(points, axis=1))
    points /= max_dist
    
    # Create new point cloud
    pcd_normalized = o3d.geometry.PointCloud()
    pcd_normalized.points = o3d.utility.Vector3dVector(points)
    
    if pcd.has_colors():
        pcd_normalized.colors = pcd.colors
    
    return pcd_normalized


def compute_point_cloud_bounds(pcd: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute axis-aligned bounding box.
    
    Args:
        pcd: Point cloud (open3d.geometry.PointCloud)
        
    Returns:
        Tuple of (min_bounds, max_bounds)
    """
    points = np.asarray(pcd.points)
    min_bounds = np.min(points, axis=0)
    max_bounds = np.max(points, axis=0)
    
    return min_bounds, max_bounds


def compute_statistics(distances: np.ndarray) -> dict:
    """
    Compute statistics of distance array.
    
    Args:
        distances: Array of distances
        
    Returns:
        Dictionary with statistics
    """
    return {
        'mean': np.mean(distances),
        'std': np.std(distances),
        'min': np.min(distances),
        'max': np.max(distances),
        'median': np.median(distances),
        'q25': np.percentile(distances, 25),
        'q75': np.percentile(distances, 75),
    }


def align_point_clouds_to_center(pcds: list) -> list:
    """
    Align multiple point clouds to their common centroid.
    
    Args:
        pcds: List of point clouds (open3d.geometry.PointCloud)
        
    Returns:
        List of centered point clouds
    """
    if not HAS_OPEN3D:
        raise ImportError("Open3D is required. Install with: pip install open3d")
    
    # Compute global centroid
    all_points = []
    for pcd in pcds:
        all_points.append(np.asarray(pcd.points))
    all_points = np.vstack(all_points)
    global_centroid = np.mean(all_points, axis=0)
    
    # Center each point cloud
    centered_pcds = []
    for pcd in pcds:
        pcd_centered = o3d.geometry.PointCloud(pcd)
        points = np.asarray(pcd_centered.points)
        points -= global_centroid
        pcd_centered.points = o3d.utility.Vector3dVector(points)
        centered_pcds.append(pcd_centered)
    
    return centered_pcds


def filter_outliers(pcd: Any,
                    nb_neighbors: int = 20,
                    std_ratio: float = 2.0) -> Any:
    """
    Remove outliers using statistical outlier removal.
    
    Args:
        pcd: Input point cloud (open3d.geometry.PointCloud)
        nb_neighbors: Number of neighbors for statistics
        std_ratio: Standard deviation ratio threshold
        
    Returns:
        Filtered point cloud
    """
    if not HAS_OPEN3D:
        raise ImportError("Open3D is required. Install with: pip install open3d")
    
    pcd_filtered, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
    return pcd_filtered


def estimate_normals(pcd: Any,
                     radius: float = 0.1) -> Any:
    """
    Estimate point cloud normals.
    
    Args:
        pcd: Input point cloud (open3d.geometry.PointCloud)
        radius: Radius for normal estimation
        
    Returns:
        Point cloud with estimated normals
    """
    if not HAS_OPEN3D:
        raise ImportError("Open3D is required. Install with: pip install open3d")
    
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamRadius(radius=radius)
    )
    return pcd


def convert_mesh_to_point_cloud(mesh: Any,
                               num_points: int = 10000) -> Any:
    """
    Convert mesh to point cloud by sampling points.
    
    Args:
        mesh: Input mesh (trimesh.Trimesh)
        num_points: Number of points to sample
        
    Returns:
        Point cloud sampled from mesh (open3d.geometry.PointCloud)
    """
    if not HAS_OPEN3D or not HAS_TRIMESH:
        raise ImportError("Open3D and Trimesh are required. Install with: pip install open3d trimesh")
    
    # Sample points from mesh surface
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    return pcd
