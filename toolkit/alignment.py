"""
ICP-based alignment for 3D point clouds and meshes.
Implements Iterative Closest Point algorithm for precise alignment.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Any

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    o3d = None


@dataclass
class AlignmentResult:
    """Result of ICP alignment."""
    transformation: np.ndarray  # 4x4 transformation matrix
    fitness: float  # Percentage of correspondences
    rmse: float  # Root Mean Square Error
    converged: bool
    num_iterations: int


class ICPAligner:
    """
    Iterative Closest Point alignment for point clouds.
    
    Supports multiple variants:
    - Point-to-Point ICP
    - Point-to-Plane ICP
    """
    
    def __init__(self, max_iterations: int = 100, 
                 threshold: float = 1e-6,
                 max_correspondence_distance: float = 0.1,
                 variant: str = "point_to_plane"):
        """
        Initialize ICP aligner.
        
        Args:
            max_iterations: Maximum number of iterations
            threshold: Convergence threshold (relative change in error)
            max_correspondence_distance: Maximum distance for correspondence
            variant: "point_to_point" or "point_to_plane"
        """
        if not HAS_OPEN3D:
            raise ImportError("Open3D is required for alignment. Install with: pip install open3d")
        
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.max_correspondence_distance = max_correspondence_distance
        self.variant = variant
        
    def align(self, source: Any, target: Any) -> AlignmentResult:
        """
        Align source point cloud to target using ICP.
        
        Args:
            source: Source point cloud (open3d.geometry.PointCloud)
            target: Target point cloud (open3d.geometry.PointCloud)
            
        Returns:
            AlignmentResult with transformation and metrics
        """
        if not HAS_OPEN3D:
            raise ImportError("Open3D is required. Install with: pip install open3d")
        if self.variant == "point_to_plane":
            reg = o3d.pipelines.registration.registration_icp(
                source, target,
                self.max_correspondence_distance,
                np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=self.max_iterations,
                    relative_fitness=self.threshold,
                    relative_rmse=self.threshold
                )
            )
        else:  # point_to_point
            reg = o3d.pipelines.registration.registration_icp(
                source, target,
                self.max_correspondence_distance,
                np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=self.max_iterations,
                    relative_fitness=self.threshold,
                    relative_rmse=self.threshold
                )
            )
        
        return AlignmentResult(
            transformation=reg.transformation,
            fitness=reg.fitness,
            rmse=reg.inlier_rmse,
            converged=reg.fitness > 0.5,
            num_iterations=len(reg.correspondence_set)
        )
    
    def transform_point_cloud(self, pcd: Any, transformation: np.ndarray) -> Any:
        """
        Apply transformation to point cloud.
        
        Args:
            pcd: Point cloud to transform (open3d.geometry.PointCloud)
            transformation: 4x4 transformation matrix
            
        Returns:
            Transformed point cloud
        """
        if not HAS_OPEN3D:
            raise ImportError("Open3D is required. Install with: pip install open3d")
        pcd_transformed = pcd.transform(transformation)
        return pcd_transformed


class MultiScaleICP:
    """
    Multi-scale ICP for coarse-to-fine alignment.
    Uses progressively finer resolutions for more robust alignment.
    """
    
    def __init__(self, scales: list = None):
        """
        Initialize multi-scale ICP.
        
        Args:
            scales: List of max correspondence distances for each scale.
                   Default: [0.5, 0.1, 0.02]
        """
        if scales is None:
            scales = [0.5, 0.1, 0.02]
        self.scales = scales
        
    def align(self, source: Any, target: Any) -> AlignmentResult:
        """
        Perform multi-scale alignment.
        
        Args:
            source: Source point cloud (open3d.geometry.PointCloud)
            target: Target point cloud (open3d.geometry.PointCloud)
            
        Returns:
            AlignmentResult from final scale
        """
        if not HAS_OPEN3D:
            raise ImportError("Open3D is required. Install with: pip install open3d")
        if not HAS_OPEN3D:
            raise ImportError("Open3D is required. Install with: pip install open3d")
        transformation = np.eye(4)
        
        for scale in self.scales:
            aligner = ICPAligner(
                max_iterations=100,
                max_correspondence_distance=scale
            )
            
            # Apply current transformation to source
            source_transformed = source.transform(transformation)
            
            # Run ICP
            result = aligner.align(source_transformed, target)
            
            # Accumulate transformation
            transformation = result.transformation @ transformation
        
        # Create final result with accumulated transformation
        final_source = source.transform(transformation)
        final_result = ICPAligner().align(final_source, target)
        final_result.transformation = transformation
        
        return final_result
