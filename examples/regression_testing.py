"""
Regression testing example - demonstrates the full regression test suite.
"""

import os
import numpy as np
from pathlib import Path
import open3d as o3d
from toolkit import RegressionTest


def create_sample_data(output_dir: str):
    """Create sample data for regression testing."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating sample data...")
    
    # Create golden reference
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    golden = sphere.sample_points_uniformly(number_of_points=1000)
    golden_path = output_dir / "golden_reference.ply"
    o3d.io.write_point_cloud(str(golden_path), golden)
    
    # Create slightly different versions (simulating test outputs)
    for i in range(3):
        test_cloud = o3d.geometry.PointCloud(golden)
        
        # Add small perturbation
        points = test_cloud.points
        points_array = points
        noise = (0.001 * (i + 1)) * np.random.randn(*points_array.shape)
        
        test_cloud = o3d.geometry.PointCloud()
        test_cloud.points = o3d.utility.Vector3dVector(points_array + noise)
        
        test_path = output_dir / f"test_output_{i}.ply"
        o3d.io.write_point_cloud(str(test_path), test_cloud)
    
    return golden_path, [
        output_dir / f"test_output_{i}.ply" for i in range(3)
    ]


def main():
    """Run regression testing example."""
    
    print("=" * 60)
    print("3D Mesh & Point-Cloud Validation Toolkit")
    print("Regression Testing Example")
    print("=" * 60)
    
    # Setup
    test_dir = Path("test_regression_data")
    test_dir.mkdir(exist_ok=True)
    
    # Create sample data
    golden_path, test_paths = create_sample_data(str(test_dir))
    
    # Initialize regression test
    print("\n1. Initializing regression test suite...")
    
    test = RegressionTest(
        golden_dir=str(test_dir),
        output_dir=str(test_dir / "results")
    )
    
    # Add quality gates
    print("\n2. Adding quality gates...")
    
    test.add_quality_gate(
        "chamfer_distance",
        "chamfer",
        threshold=0.01,
        operator="<="
    )
    
    test.add_quality_gate(
        "hausdorff_distance",
        "hausdorff",
        threshold=0.05,
        operator="<="
    )
    
    print("   - Chamfer distance <= 0.01")
    print("   - Hausdorff distance <= 0.05")
    
    # Run regression tests
    print("\n3. Running regression tests...")
    
    results = []
    for i, test_path in enumerate(test_paths):
        print(f"\n   Test {i+1}: {test_path.name}")
        result = test.run(
            f"test_{i+1}",
            str(test_path),
            str(golden_path)
        )
        
        print(f"   - Chamfer Distance: {result.metrics.get('chamfer_distance', 'N/A'):.6f}")
        print(f"   - Hausdorff Distance: {result.metrics.get('hausdorff_distance', 'N/A'):.6f}")
        print(f"   - Status: {'PASS' if result.passed else 'FAIL'}")
        
        results.append(result)
    
    # Generate summary
    print("\n4. Generating test summary...")
    
    summary = test.generate_summary(results)
    
    print(f"\n   Total Tests: {summary['total_tests']}")
    print(f"   Passed: {summary['passed_tests']}")
    print(f"   Failed: {summary['failed_tests']}")
    print(f"   Pass Rate: {summary['pass_rate']*100:.1f}%")
    
    # Save summary
    summary_file = test.save_summary(results, "regression_summary.json")
    
    print(f"\n5. Saved summary to: {summary_file}")
    
    # Print metric statistics
    print("\n6. Metric Statistics:")
    
    for metric_name, stats in summary['metric_statistics'].items():
        print(f"\n   {metric_name}:")
        for stat_name, stat_value in stats.items():
            print(f"      {stat_name}: {stat_value:.6f}")
    
    print("\n" + "=" * 60)
    print("Regression testing completed!")
    print("=" * 60)
    
    # Cleanup
    import shutil
    if test_dir.exists():
        shutil.rmtree(test_dir)
        print("\nCleaned up test data.")


if __name__ == "__main__":
    main()
