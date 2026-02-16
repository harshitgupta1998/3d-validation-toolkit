"""
Basic alignment example - demonstrates ICP alignment and metrics.
"""

import numpy as np
import open3d as o3d
from toolkit import ICPAligner, ChamferDistance, HausdorffDistance


def main():
    """Run basic alignment example."""
    
    print("=" * 60)
    print("3D Mesh & Point-Cloud Validation Toolkit")
    print("Basic Alignment Example")
    print("=" * 60)
    
    # Create sample point clouds
    print("\n1. Creating sample point clouds...")
    
    # Create a sphere as source
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    source = sphere.sample_points_uniformly(number_of_points=1000)
    source.paint_uniform_color([0, 1, 0])  # Green
    
    # Create target by transforming source
    target = o3d.geometry.PointCloud(source)
    
    # Apply transformation: rotation + translation
    R = o3d.geometry.get_rotation_matrix_from_xyz(
        [np.pi/6, np.pi/4, np.pi/3]
    )
    t = np.array([0.5, 0.3, -0.2])
    transformation = np.eye(4)
    transformation[:3, :3] = R
    transformation[:3, 3] = t
    
    target = target.transform(transformation)
    target.paint_uniform_color([1, 0, 0])  # Red
    
    print(f"Source points: {len(np.asarray(source.points))}")
    print(f"Target points: {len(np.asarray(target.points))}")
    
    # Compute initial metrics
    print("\n2. Computing initial metrics (before alignment)...")
    
    chamfer = ChamferDistance()
    hausdorff = HausdorffDistance()
    
    initial_chamfer = chamfer.compute(source, target)
    initial_hausdorff = hausdorff.compute(source, target)
    
    print(f"Initial Chamfer Distance: {initial_chamfer:.6f}")
    print(f"Initial Hausdorff Distance: {initial_hausdorff:.6f}")
    
    # Perform ICP alignment
    print("\n3. Performing ICP alignment...")
    
    aligner = ICPAligner(
        max_iterations=100,
        max_correspondence_distance=0.1,
        variant="point_to_plane"
    )
    
    result = aligner.align(target, source)
    
    print(f"Fitness: {result.fitness:.4f}")
    print(f"RMSE: {result.rmse:.6f}")
    print(f"Converged: {result.converged}")
    
    # Apply transformation to target
    aligned_target = target.transform(result.transformation)
    aligned_target.paint_uniform_color([0, 0, 1])  # Blue
    
    # Compute final metrics
    print("\n4. Computing final metrics (after alignment)...")
    
    final_chamfer = chamfer.compute(source, aligned_target)
    final_hausdorff = hausdorff.compute(source, aligned_target)
    
    print(f"Final Chamfer Distance: {final_chamfer:.6f}")
    print(f"Final Hausdorff Distance: {final_hausdorff:.6f}")
    
    # Print improvement
    print("\n5. Improvement Summary:")
    print(f"Chamfer Distance reduced by: {initial_chamfer - final_chamfer:.6f} ({(initial_chamfer - final_chamfer) / initial_chamfer * 100:.1f}%)")
    print(f"Hausdorff Distance reduced by: {initial_hausdorff - final_hausdorff:.6f} ({(initial_hausdorff - final_hausdorff) / initial_hausdorff * 100:.1f}%)")
    
    # Print transformation matrix
    print("\n6. Estimated Transformation Matrix:")
    print(result.transformation)
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    
    # Optional: visualize (uncomment to visualize)
    # print("\nVisualizing alignment...")
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.add_geometry(source)
    # vis.add_geometry(target)
    # vis.add_geometry(aligned_target)
    # vis.run()
    # vis.destroy_window()


if __name__ == "__main__":
    main()
