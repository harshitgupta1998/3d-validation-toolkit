# 3D Mesh & Point-Cloud Validation Toolkit

A comprehensive Python toolkit for validating 3D mesh and point-cloud data with advanced alignment, geometric error metrics, and automated regression testing.

## Features

- **ICP-Based Alignment**: Iterative Closest Point (ICP) algorithm for precise 3D shape alignment
- **Geometric Error Metrics**:
  - Chamfer Distance: Symmetric bidirectional distance between point clouds
  - Point-to-Surface Distance: Measure distance from points to mesh surfaces
- **Visual Overlays**: Generate 3D visualizations comparing predicted vs. golden assets
- **Regression Test Suite**: Automated testing with golden assets and quality gates
- **Auto-Generated Reports**: HTML/JSON reports tracking precision drift across versions
- **Threshold-Based Quality Gates**: Validate output against configurable metrics

## Requirements

- Python 3.8+
- Open3D >= 0.13
- trimesh >= 3.9
- NumPy >= 1.19
- PyTest >= 6.0
- matplotlib >= 3.3
- scikit-learn >= 0.24

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Basic Alignment and Error Metrics

```python
from toolkit.alignment import ICPAligner
from toolkit.metrics import ChamferDistance, PointToSurfaceDistance

# Load point clouds
source = open3d_pcd  # Your source point cloud
target = open3d_pcd  # Your target point cloud

# Align using ICP
aligner = ICPAligner(max_iterations=100)
result = aligner.align(source, target)

print(f"Fitness: {result.fitness}")
print(f"RMSE: {result.rmse}")

# Calculate metrics
chamfer = ChamferDistance()
distance = chamfer.compute(source, target)
print(f"Chamfer Distance: {distance}")
```

### 2. Regression Testing

```python
from toolkit.regression import RegressionTest

test = RegressionTest(golden_dir="golden_assets", output_dir="results")
test.add_quality_gate("chamfer_distance", threshold=0.05)
test.add_quality_gate("point_to_surface", threshold=0.10)

result = test.run("test_mesh.ply", "predicted_mesh.ply")
print(f"Test Passed: {result.passed}")
```

### 3. Generate Comparison Report

```python
from toolkit.reporting import ComparisonReporter

reporter = ComparisonReporter(output_dir="reports")
report = reporter.generate_report(
    source_mesh="golden.ply",
    target_mesh="predicted.ply",
    alignment_result=result,
    metrics=metrics_dict
)
print(f"Report saved to {report.filepath}")
```

## Project Structure

```
toolkit/
├── alignment.py          # ICP alignment implementation
├── metrics.py            # Geometric error metrics (Chamfer, P2S)
├── visualization.py      # 3D visualization and overlays
├── regression.py         # Regression test framework
├── reporting.py          # HTML/JSON report generation
└── utils.py             # Utility functions

tests/
├── test_alignment.py     # ICP alignment tests
├── test_metrics.py       # Metrics calculation tests
├── test_regression.py    # Regression suite tests
└── golden_assets/        # Reference data for regression tests

examples/
├── basic_alignment.py    # Basic usage example
├── regression_testing.py # Full regression test example
└── visualization.py      # Visualization example

requirements.txt          # Python dependencies
setup.py                 # Package setup
```

## Usage Examples

See the `examples/` directory for complete examples.

## Testing

Run the regression test suite:

```bash
pytest tests/ -v
```

Run specific tests:

```bash
pytest tests/test_alignment.py -v
pytest tests/test_metrics.py -v
```

## Documentation

- [Alignment Guide](docs/alignment.md)
- [Metrics Reference](docs/metrics.md)
- [Regression Testing](docs/regression_testing.md)
- [Report Generation](docs/reporting.md)

## License

MIT

## Contributing

Contributions are welcome! Please ensure:

- All tests pass
- Code follows PEP 8 style guide
- New features include tests and documentation
