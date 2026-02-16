"""
Visualization example - demonstrates mesh visualization and comparison.
"""

import numpy as np
import open3d as o3d
from toolkit import MeshVisualizer, ComparisonOverlay, ChamferDistance


def main():
    """Run visualization example."""
    
    print("=" * 60)
    print("3D Mesh & Point-Cloud Validation Toolkit")
    print("Visualization Example")
    print("=" * 60)
    
    # Create sample point clouds
    print("\n1. Creating sample point clouds...")
    
    # Source cloud
    sphere1 = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    source = sphere1.sample_points_uniformly(number_of_points=1000)
    
    # Target cloud with noise
    target = o3d.geometry.PointCloud(source)
    target_points = np.asarray(target.points)
    noise = 0.02 * np.random.randn(*target_points.shape)
    target.points = o3d.utility.Vector3dVector(target_points + noise)
    
    print(f"Source points: {len(np.asarray(source.points))}")
    print(f"Target points: {len(np.asarray(target.points))}")
    
    # Compute distances
    print("\n2. Computing distances...")
    
    chamfer = ChamferDistance()
    detailed_metrics = chamfer.compute_detailed(source, target)
    
    print(f"Chamfer Distance: {detailed_metrics['chamfer_distance']:.6f}")
    print(f"Mean S2T: {detailed_metrics['s2t_mean']:.6f}")
    print(f"Mean T2S: {detailed_metrics['t2s_mean']:.6f}")
    print(f"Max S2T: {detailed_metrics['s2t_max']:.6f}")
    print(f"Max T2S: {detailed_metrics['t2s_max']:.6f}")
    
    # Compute point-to-point distances for visualization
    print("\n3. Creating colored visualizations...")
    
    from scipy.spatial import cKDTree
    
    source_np = np.asarray(source.points)
    target_np = np.asarray(target.points)
    
    target_tree = cKDTree(target_np)
    distances, _ = target_tree.query(source_np)
    
    # Create colored point cloud
    source_colored = MeshVisualizer.create_colored_pcd(
        source, distances, colormap='hot'
    )
    
    print("Created colored point cloud (hot colormap)")
    
    # Create plots
    print("\n4. Creating comparison plots...")
    
    # Error histogram
    hist_fig = ComparisonOverlay.create_error_histogram(
        distances,
        title="Error Distribution (P2P Distances)",
        bins=50
    )
    print("Created error histogram")
    
    # Save figures
    hist_fig.savefig("error_histogram.png", dpi=150)
    print("Saved error_histogram.png")
    
    # Metrics plot
    metrics_dict = {
        'chamfer': detailed_metrics['chamfer_distance'],
        'mean_s2t': detailed_metrics['s2t_mean'],
        'mean_t2s': detailed_metrics['t2s_mean'],
    }
    
    metrics_fig = ComparisonOverlay.create_statistics_plot(metrics_dict)
    print("Created metrics plot")
    
    # Save metrics plot
    metrics_fig.savefig("metrics_plot.png", dpi=150)
    print("Saved metrics_plot.png")
    
    print("\n" + "=" * 60)
    print("Visualization example completed!")
    print("Output saved to: error_histogram.png, metrics_plot.png")
    print("=" * 60)
    
    # Optional: Interactive visualization (uncomment to use)
    # print("\nStarting interactive visualization...")
    # print("Close window to continue.")
    # MeshVisualizer.visualize_comparison(source, target, distances)


if __name__ == "__main__":
    main()
