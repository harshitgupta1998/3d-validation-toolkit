"""
Geometric error metrics for 3D mesh and point cloud comparison.
Implements Chamfer distance, point-to-surface distance, and other metrics.
"""

from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple, Any

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    o3d = None

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False
    trimesh = None


class ChamferDistance:
    """
    Chamfer distance between two point clouds.
    
    Measures the symmetric average of nearest-neighbor distances
    in both directions.
    """
    
    def compute(self, source: Any, target: Any) -> float:
        """
        Compute Chamfer distance.
        
        Args:
            source: Source point cloud (open3d.geometry.PointCloud)
            target: Target point cloud (open3d.geometry.PointCloud)
            
        Returns:
            Chamfer distance value
        """
        source_np = np.asarray(source.points)
        target_np = np.asarray(target.points)
        
        # Build KD-trees
        source_tree = cKDTree(source_np)
        target_tree = cKDTree(target_np)
        
        # Distances from source to target
        distances_s2t, _ = source_tree.query(target_np)
        d_s2t = np.mean(distances_s2t)
        
        # Distances from target to source
        distances_t2s, _ = target_tree.query(source_np)
        d_t2s = np.mean(distances_t2s)
        
        # Chamfer distance
        chamfer = (d_s2t + d_t2s) / 2.0
        
        return chamfer
    
    def compute_detailed(self, source: Any, target: Any) -> dict:
        """
        Compute Chamfer distance with detailed statistics.
        
        Args:
            source: Source point cloud (open3d.geometry.PointCloud)
            target: Target point cloud (open3d.geometry.PointCloud)
            
        Returns:
            Dictionary with detailed metrics
        """
        source_np = np.asarray(source.points)
        target_np = np.asarray(target.points)
        
        source_tree = cKDTree(source_np)
        target_tree = cKDTree(target_np)
        
        distances_s2t, _ = source_tree.query(target_np)
        distances_t2s, _ = target_tree.query(source_np)
        
        return {
            "chamfer_distance": (np.mean(distances_s2t) + np.mean(distances_t2s)) / 2.0,
            "s2t_mean": np.mean(distances_s2t),
            "s2t_std": np.std(distances_s2t),
            "s2t_max": np.max(distances_s2t),
            "t2s_mean": np.mean(distances_t2s),
            "t2s_std": np.std(distances_t2s),
            "t2s_max": np.max(distances_t2s),
        }


class PointToSurfaceDistance:
    """
    Point-to-surface distance between point cloud and mesh.
    
    Measures the minimum distance from points to the surface of a mesh.
    """
    
    def __init__(self, use_signed_distance: bool = False):
        """
        Initialize metric.
        
        Args:
            use_signed_distance: Whether to use signed distance (requires closed mesh)
        """
        self.use_signed_distance = use_signed_distance
    
    def compute(self, points: np.ndarray, mesh: Any) -> float:
        """
        Compute point-to-surface distance.
        
        Args:
            points: Point cloud as numpy array (N, 3)
            mesh: Target mesh (trimesh.Trimesh)
            
        Returns:
            Mean point-to-surface distance
        """
        if self.use_signed_distance and mesh.is_watertight:
            distances = mesh.signed_distance(points)
            return np.mean(np.abs(distances))
        else:
            # Unsigned distance: distance to nearest point on mesh
            distances, _ = trimesh.proximity.closest_point(mesh, points)
            distances = np.linalg.norm(points - distances, axis=1)
            return np.mean(distances)
    
    def compute_detailed(self, points: np.ndarray, mesh: Any) -> dict:
        """
        Compute point-to-surface distance with statistics.
        
        Args:
            points: Point cloud as numpy array (N, 3)
            mesh: Target mesh
            
        Returns:
            Dictionary with detailed metrics
        """
        closest_points, distances = trimesh.proximity.closest_point(mesh, points)
        distances = np.linalg.norm(points - closest_points, axis=1)
        
        return {
            "p2s_distance": np.mean(distances),
            "p2s_std": np.std(distances),
            "p2s_max": np.max(distances),
            "p2s_min": np.min(distances),
            "p2s_median": np.median(distances),
        }


class HausdorffDistance:
    """
    Hausdorff distance between two point clouds.
    
    Maximum of the minimum distances (most distant point pair).
    """
    
    def compute(self, source: Any, target: Any) -> float:
        """
        Compute Hausdorff distance.
        
        Args:
            source: Source point cloud (open3d.geometry.PointCloud)
            target: Target point cloud (open3d.geometry.PointCloud)
            
        Returns:
            Hausdorff distance
        """
        source_np = np.asarray(source.points)
        target_np = np.asarray(target.points)
        
        source_tree = cKDTree(source_np)
        target_tree = cKDTree(target_np)
        
        # Maximum distance from target to source
        distances_t2s, _ = target_tree.query(source_np)
        d_t2s = np.max(distances_t2s)
        
        # Maximum distance from source to target
        distances_s2t, _ = source_tree.query(target_np)
        d_s2t = np.max(distances_s2t)
        
        # Hausdorff distance
        hausdorff = max(d_s2t, d_t2s)
        
        return hausdorff
    
    def compute_directed(self, source: Any, target: Any) -> Tuple[float, float]:
        """
        Compute directed Hausdorff distances.
        
        Args:
            source: Source point cloud
            target: Target point cloud
            
        Returns:
            Tuple of (s2t_hausdorff, t2s_hausdorff)
        """
        source_np = np.asarray(source.points)
        target_np = np.asarray(target.points)
        
        source_tree = cKDTree(source_np)
        target_tree = cKDTree(target_np)
        
        distances_s2t, _ = source_tree.query(target_np)
        d_s2t = np.max(distances_s2t)
        
        distances_t2s, _ = target_tree.query(source_np)
        d_t2s = np.max(distances_t2s)
        
        return d_s2t, d_t2s


class VolumeError:
    """
    Volume error between two meshes.
    Useful for volumetric comparisons.
    """
    
    def compute(self, mesh1: Any, mesh2: Any) -> float:
        """
        Compute volume difference.
        
        Args:
            mesh1: First mesh (trimesh.Trimesh, must be watertight)
            mesh2: Second mesh (trimesh.Trimesh, must be watertight)
            
        Returns:
            Absolute volume difference
        """
        if not (mesh1.is_watertight and mesh2.is_watertight):
            raise ValueError("Both meshes must be watertight for volume comparison")
        
        vol_diff = abs(mesh1.volume - mesh2.volume)
        return vol_diff


class SurfaceAreaError:
    """
    Surface area error between two meshes.
    """
    
    def compute(self, mesh1: Any, mesh2: Any) -> float:
        """
        Compute surface area difference.
        
        Args:
            mesh1: First mesh
            mesh2: Second mesh
            
        Returns:
            Absolute surface area difference
        """
        area_diff = abs(mesh1.area - mesh2.area)
        return area_diff
