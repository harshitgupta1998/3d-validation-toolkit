"""
3D visualization and comparison overlays for meshes and point clouds.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import List, Tuple, Optional, Any

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


class MeshVisualizer:
    """
    Visualize meshes and point clouds with various coloring schemes.
    """
    
    @staticmethod
    def create_colored_pcd(pcd: Any, distances: np.ndarray,
                          colormap: str = 'viridis') -> Any:
        """
        Create colored point cloud based on distance values.
        
        Args:
            pcd: Point cloud to color (open3d.geometry.PointCloud)
            distances: Distance values for each point
            colormap: Matplotlib colormap name
            
        Returns:
            Colored point cloud
        """
        if not HAS_OPEN3D:
            raise ImportError("Open3D is required. Install with: pip install open3d")
        cmap = cm.get_cmap(colormap)
        
        # Normalize distances to [0, 1]
        distances_min = np.min(distances)
        distances_max = np.max(distances)
        if distances_max > distances_min:
            normalized = (distances - distances_min) / (distances_max - distances_min)
        else:
            normalized = np.zeros_like(distances)
        
        # Map to colors
        colors = cmap(normalized)[:, :3]
        
        pcd_colored = o3d.geometry.PointCloud(pcd)
        pcd_colored.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd_colored
    
    @staticmethod
    def visualize_comparison(source: Any, target: Any,
                            distances: np.ndarray = None,
                            window_name: str = "Comparison Visualization"):
        """
        Visualize source and target point clouds side by side.
        
        Args:
            source: Source point cloud (open3d.geometry.PointCloud)
            target: Target point cloud (open3d.geometry.PointCloud)
            distances: Optional distance values for coloring source
            window_name: Window title
        """
        if not HAS_OPEN3D:
            raise ImportError("Open3D is required. Install with: pip install open3d")
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name, width=1400, height=700)
        
        # Color source by distance if provided
        if distances is not None:
            source_vis = MeshVisualizer.create_colored_pcd(source, distances)
        else:
            source_vis = source
            source_vis.paint_uniform_color([0.1, 0.9, 0.1])  # Green
        
        # Color target
        target_vis = target
        target_vis.paint_uniform_color([0.9, 0.1, 0.1])  # Red
        
        vis.add_geometry(source_vis)
        vis.add_geometry(target_vis)
        
        # Set view
        ctr = vis.get_view_control()
        ctr.reset_camera()
        
        vis.run()
        vis.destroy_window()
    
    @staticmethod
    def visualize_error_heatmap(pcd: Any, distances: np.ndarray,
                               title: str = "Error Heatmap",
                               colormap: str = 'hot'):
        """
        Create heatmap visualization of errors.
        
        Args:
            pcd: Point cloud (open3d.geometry.PointCloud)
            distances: Distance/error values
            title: Plot title
            colormap: Color map to use
        """
        if not HAS_OPEN3D:
            raise ImportError("Open3D is required. Install with: pip install open3d")
        pcd_colored = MeshVisualizer.create_colored_pcd(pcd, distances, colormap)
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title)
        vis.add_geometry(pcd_colored)
        vis.run()
        vis.destroy_window()


class ComparisonOverlay:
    """
    Create side-by-side comparison visualizations.
    """
    
    @staticmethod
    def create_alignment_overlay(source: Any,
                                target: Any,
                                aligned_source: Any,
                                output_path: str = None):
        """
        Create visualization showing before and after alignment.
        
        Args:
            source: Original source point cloud
            target: Target point cloud
            aligned_source: Aligned source point cloud
            output_path: Optional path to save visualization
        """
        # Create visualization
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Alignment Overlay", width=1400, height=700)
        
        # Color clouds
        source.paint_uniform_color([1.0, 0.0, 0.0])  # Red - unaligned
        aligned_source.paint_uniform_color([0.0, 1.0, 0.0])  # Green - aligned
        target.paint_uniform_color([0.0, 0.0, 1.0])  # Blue - target
        
        vis.add_geometry(source)
        vis.add_geometry(aligned_source)
        vis.add_geometry(target)
        
        # Auto view
        ctr = vis.get_view_control()
        ctr.reset_camera()
        
        vis.run()
        vis.destroy_window()
        
        if output_path:
            vis.capture_screen_image(output_path)
    
    @staticmethod
    def create_error_histogram(distances: np.ndarray,
                              title: str = "Error Distribution",
                              bins: int = 50,
                              output_path: str = None) -> plt.Figure:
        """
        Create histogram of errors.
        
        Args:
            distances: Array of distance/error values
            title: Plot title
            bins: Number of histogram bins
            output_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(distances, bins=bins, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Distance/Error')
        ax.set_ylabel('Count')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        stats_text = (
            f"Mean: {np.mean(distances):.4f}\n"
            f"Std: {np.std(distances):.4f}\n"
            f"Max: {np.max(distances):.4f}\n"
            f"Min: {np.min(distances):.4f}"
        )
        ax.text(0.70, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=150)
        
        return fig
    
    @staticmethod
    def create_statistics_plot(metrics_dict: dict,
                              output_path: str = None) -> plt.Figure:
        """
        Create bar plot of metrics.
        
        Args:
            metrics_dict: Dictionary of metric names and values
            output_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        keys = list(metrics_dict.keys())
        values = list(metrics_dict.values())
        
        bars = ax.bar(range(len(keys)), values, color='steelblue', alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom')
        
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys, rotation=45, ha='right')
        ax.set_ylabel('Value')
        ax.set_title('Quality Metrics')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=150)
        
        return fig
