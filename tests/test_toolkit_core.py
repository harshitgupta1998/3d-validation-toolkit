"""
Core toolkit tests that don't require Open3D.
Tests the structure, initialization, and configuration.
"""

import pytest
from pathlib import Path
import tempfile
import json
from toolkit.regression import RegressionTest, QualityGate, RegressionTestResult


class TestQualityGate:
    """Test QualityGate class."""
    
    def test_gate_initialization(self):
        """Test QualityGate initialization."""
        gate = QualityGate(
            name="test_gate",
            metric_type="chamfer",
            threshold=0.05,
            operator="<="
        )
        
        assert gate.name == "test_gate"
        assert gate.metric_type == "chamfer"
        assert gate.threshold == 0.05
        assert gate.operator == "<="
    
    def test_gate_with_less_than_operator(self):
        """Test < operator."""
        gate = QualityGate(
            name="test",
            metric_type="chamfer",
            threshold=0.05,
            operator="<"
        )
        
        assert gate.operator == "<"
    
    def test_gate_with_greater_than_operator(self):
        """Test > operator."""
        gate = QualityGate(
            name="test",
            metric_type="hausdorff",
            threshold=0.1,
            operator=">"
        )
        
        assert gate.operator == ">"
    
    def test_gate_with_equal_operator(self):
        """Test == operator."""
        gate = QualityGate(
            name="test",
            metric_type="volume",
            threshold=100.0,
            operator="=="
        )
        
        assert gate.operator == "=="


class TestRegressionTestResult:
    """Test RegressionTestResult class."""
    
    def test_result_initialization(self):
        """Test result initialization."""
        result = RegressionTestResult(
            test_name="test1",
            passed=True,
            metrics={"chamfer": 0.03},
            quality_gates={"gate1": True}
        )
        
        assert result.test_name == "test1"
        assert result.passed == True
        assert result.metrics["chamfer"] == 0.03
        assert result.quality_gates["gate1"] == True
    
    def test_result_with_error_message(self):
        """Test result with error message."""
        result = RegressionTestResult(
            test_name="test1",
            passed=False,
            metrics={},
            quality_gates={},
            error_message="Test failed due to missing file"
        )
        
        assert result.passed == False
        assert result.error_message == "Test failed due to missing file"
    
    def test_result_with_timestamp(self):
        """Test result with timestamp."""
        result = RegressionTestResult(
            test_name="test1",
            passed=True,
            metrics={"chamfer": 0.03},
            quality_gates={},
            timestamp="2026-02-15T12:00:00"
        )
        
        assert result.timestamp == "2026-02-15T12:00:00"


class TestRegressionTestInitialization:
    """Test RegressionTest initialization."""
    
    def test_regression_test_init(self):
        """Test RegressionTest initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test = RegressionTest(
                golden_dir=tmpdir,
                output_dir=Path(tmpdir) / "results"
            )
            
            assert test.golden_dir == Path(tmpdir)
            assert (Path(tmpdir) / "results").exists()
    
    def test_regression_test_creates_output_dir(self):
        """Test that output directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_results"
            test = RegressionTest(
                golden_dir=tmpdir,
                output_dir=output_path
            )
            
            assert output_path.exists()
            assert output_path.is_dir()
    
    def test_add_single_quality_gate(self):
        """Test adding a single quality gate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test = RegressionTest(golden_dir=tmpdir)
            
            test.add_quality_gate(
                name="chamfer_test",
                metric_type="chamfer",
                threshold=0.05,
                operator="<="
            )
            
            assert "chamfer_test" in test.quality_gates
            gate = test.quality_gates["chamfer_test"]
            assert gate.threshold == 0.05
            assert gate.operator == "<="
    
    def test_add_multiple_quality_gates(self):
        """Test adding multiple quality gates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test = RegressionTest(golden_dir=tmpdir)
            
            test.add_quality_gate("gate1", "chamfer", 0.05, "<=")
            test.add_quality_gate("gate2", "hausdorff", 0.1, "<=")
            test.add_quality_gate("gate3", "p2s", 0.08, "<=")
            
            assert len(test.quality_gates) == 3
            assert "gate1" in test.quality_gates
            assert "gate2" in test.quality_gates
            assert "gate3" in test.quality_gates
    
    def test_gate_overwrite(self):
        """Test that adding a gate with same name overwrites."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test = RegressionTest(golden_dir=tmpdir)
            
            test.add_quality_gate("gate1", "chamfer", 0.05, "<=")
            test.add_quality_gate("gate1", "chamfer", 0.1, "<=")
            
            assert len(test.quality_gates) == 1
            assert test.quality_gates["gate1"].threshold == 0.1


class TestQualityGateEvaluation:
    """Test quality gate evaluation logic."""
    
    def test_check_gate_less_than_or_equal_pass(self):
        """Test <= operator evaluation - pass case."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test = RegressionTest(golden_dir=tmpdir)
            test.add_quality_gate("test", "chamfer", 0.05, "<=")
            
            gate = test.quality_gates["test"]
            
            assert test._check_gate(gate, 0.03) == True
            assert test._check_gate(gate, 0.05) == True
    
    def test_check_gate_less_than_or_equal_fail(self):
        """Test <= operator evaluation - fail case."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test = RegressionTest(golden_dir=tmpdir)
            test.add_quality_gate("test", "chamfer", 0.05, "<=")
            
            gate = test.quality_gates["test"]
            
            assert test._check_gate(gate, 0.07) == False
            assert test._check_gate(gate, 0.1) == False
    
    def test_check_gate_less_than(self):
        """Test < operator evaluation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test = RegressionTest(golden_dir=tmpdir)
            test.add_quality_gate("test", "chamfer", 0.05, "<")
            
            gate = test.quality_gates["test"]
            
            assert test._check_gate(gate, 0.03) == True
            assert test._check_gate(gate, 0.05) == False
            assert test._check_gate(gate, 0.07) == False
    
    def test_check_gate_greater_than_or_equal(self):
        """Test >= operator evaluation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test = RegressionTest(golden_dir=tmpdir)
            test.add_quality_gate("test", "fitness", 0.9, ">=")
            
            gate = test.quality_gates["test"]
            
            assert test._check_gate(gate, 0.95) == True
            assert test._check_gate(gate, 0.90) == True
            assert test._check_gate(gate, 0.85) == False
    
    def test_check_gate_greater_than(self):
        """Test > operator evaluation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test = RegressionTest(golden_dir=tmpdir)
            test.add_quality_gate("test", "fitness", 0.9, ">")
            
            gate = test.quality_gates["test"]
            
            assert test._check_gate(gate, 0.95) == True
            assert test._check_gate(gate, 0.90) == False
            assert test._check_gate(gate, 0.85) == False
    
    def test_check_gate_equal(self):
        """Test == operator evaluation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test = RegressionTest(golden_dir=tmpdir)
            test.add_quality_gate("test", "status", 1, "==")
            
            gate = test.quality_gates["test"]
            
            assert test._check_gate(gate, 1) == True
            assert test._check_gate(gate, 0) == False


class TestToolkitImports:
    """Test that all toolkit modules can be imported."""
    
    def test_import_alignment_module(self):
        """Test alignment module import."""
        from toolkit import alignment
        assert hasattr(alignment, 'ICPAligner')
        assert hasattr(alignment, 'MultiScaleICP')
        assert hasattr(alignment, 'AlignmentResult')
    
    def test_import_alignment_classes(self):
        """Test importing alignment classes directly."""
        from toolkit import ICPAligner, MultiScaleICP, AlignmentResult
        assert ICPAligner is not None
        assert MultiScaleICP is not None
        assert AlignmentResult is not None
    
    def test_import_metrics_module(self):
        """Test metrics module import."""
        from toolkit import metrics
        assert hasattr(metrics, 'ChamferDistance')
        assert hasattr(metrics, 'HausdorffDistance')
        assert hasattr(metrics, 'PointToSurfaceDistance')
        assert hasattr(metrics, 'VolumeError')
        assert hasattr(metrics, 'SurfaceAreaError')
    
    def test_import_metrics_classes(self):
        """Test importing metric classes directly."""
        from toolkit import ChamferDistance, HausdorffDistance, PointToSurfaceDistance
        assert ChamferDistance is not None
        assert HausdorffDistance is not None
        assert PointToSurfaceDistance is not None
    
    def test_import_regression_module(self):
        """Test regression module import."""
        from toolkit import regression
        assert hasattr(regression, 'RegressionTest')
        assert hasattr(regression, 'QualityGate')
        assert hasattr(regression, 'RegressionTestResult')
    
    def test_import_regression_classes(self):
        """Test importing regression classes directly."""
        from toolkit import RegressionTest, QualityGate, RegressionTestResult
        assert RegressionTest is not None
        assert QualityGate is not None
        assert RegressionTestResult is not None
    
    def test_import_utils_functions(self):
        """Test utils module import."""
        from toolkit import utils
        assert callable(utils.load_point_cloud)
        assert callable(utils.load_mesh)
        assert callable(utils.save_point_cloud)
        assert callable(utils.downsample_point_cloud)
        assert callable(utils.normalize_point_cloud)
        assert callable(utils.compute_point_cloud_bounds)
        assert callable(utils.compute_statistics)
        assert callable(utils.filter_outliers)
        assert callable(utils.estimate_normals)
        assert callable(utils.convert_mesh_to_point_cloud)
    
    def test_import_utils_directly(self):
        """Test importing utils functions directly."""
        from toolkit import (
            load_point_cloud, load_mesh, save_point_cloud,
            downsample_point_cloud, normalize_point_cloud,
            compute_point_cloud_bounds, compute_statistics
        )
        assert callable(load_point_cloud)
        assert callable(load_mesh)
        assert callable(save_point_cloud)
    
    def test_import_visualization_module(self):
        """Test visualization module import."""
        from toolkit import visualization
        assert hasattr(visualization, 'MeshVisualizer')
        assert hasattr(visualization, 'ComparisonOverlay')
    
    def test_import_visualization_classes(self):
        """Test importing visualization classes directly."""
        from toolkit import MeshVisualizer, ComparisonOverlay
        assert MeshVisualizer is not None
        assert ComparisonOverlay is not None
    
    def test_import_reporting_module(self):
        """Test reporting module import."""
        from toolkit import reporting
        assert hasattr(reporting, 'ComparisonReporter')
    
    def test_import_reporting_classes(self):
        """Test importing reporting classes directly."""
        from toolkit import ComparisonReporter
        assert ComparisonReporter is not None


class TestUtilityFunctions:
    """Test utility function signatures and basic properties."""
    
    def test_compute_statistics_function(self):
        """Test that compute_statistics is callable."""
        from toolkit.utils import compute_statistics
        import numpy as np
        
        # Test with sample data
        distances = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        stats = compute_statistics(distances)
        
        assert isinstance(stats, dict)
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'median' in stats
        assert 'q25' in stats
        assert 'q75' in stats
    
    def test_compute_statistics_values(self):
        """Test that compute_statistics returns correct values."""
        from toolkit.utils import compute_statistics
        import numpy as np
        
        distances = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = compute_statistics(distances)
        
        assert stats['mean'] == 3.0
        assert stats['min'] == 1.0
        assert stats['max'] == 5.0
        assert stats['median'] == 3.0


class TestRegressionTestStructure:
    """Test RegressionTest internal structure."""
    
    def test_regression_test_has_golden_dir(self):
        """Test that RegressionTest stores golden_dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test = RegressionTest(golden_dir=tmpdir)
            assert test.golden_dir == Path(tmpdir)
    
    def test_regression_test_has_output_dir(self):
        """Test that RegressionTest stores output_dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "out"
            test = RegressionTest(golden_dir=tmpdir, output_dir=output)
            assert test.output_dir == output
    
    def test_regression_test_has_quality_gates_dict(self):
        """Test that RegressionTest has quality_gates dictionary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test = RegressionTest(golden_dir=tmpdir)
            assert isinstance(test.quality_gates, dict)
    
    def test_regression_test_has_add_quality_gate_method(self):
        """Test that RegressionTest has add_quality_gate method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test = RegressionTest(golden_dir=tmpdir)
            assert callable(test.add_quality_gate)
    
    def test_regression_test_has_check_gate_method(self):
        """Test that RegressionTest has _check_gate method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test = RegressionTest(golden_dir=tmpdir)
            assert callable(test._check_gate)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
