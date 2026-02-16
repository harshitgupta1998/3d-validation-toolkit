"""
Regression testing framework with golden assets and quality gates.
"""

import os
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

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

from .alignment import ICPAligner
from .metrics import ChamferDistance, PointToSurfaceDistance, HausdorffDistance


@dataclass
class QualityGate:
    """Represents a quality gate constraint."""
    name: str
    metric_type: str  # 'chamfer', 'p2s', 'hausdorff', 'volume', etc.
    threshold: float
    operator: str = '<='  # '<', '<=', '>', '>=', '=='


@dataclass
class RegressionTestResult:
    """Result of a regression test."""
    test_name: str
    passed: bool
    metrics: Dict[str, float]
    quality_gates: Dict[str, bool]
    error_message: Optional[str] = None
    timestamp: Optional[str] = None


class RegressionTest:
    """
    Regression testing framework with golden assets and quality gates.
    
    Example:
        test = RegressionTest(golden_dir="golden_assets")
        test.add_quality_gate("chamfer_distance", "<", 0.05)
        result = test.run("model.ply", "output.ply")
    """
    
    def __init__(self, golden_dir: str = "golden_assets",
                 output_dir: str = "test_results",
                 metadata_file: str = "metadata.json"):
        """
        Initialize regression test suite.
        
        Args:
            golden_dir: Directory containing golden reference assets
            output_dir: Directory to save test results
            metadata_file: Name of metadata file in golden_dir
        """
        self.golden_dir = Path(golden_dir)
        self.output_dir = Path(output_dir)
        self.metadata_file = metadata_file
        
        # Create directories if they don't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Quality gates
        self.quality_gates: Dict[str, QualityGate] = {}
        
        # Metrics calculators
        self.chamfer_calc = ChamferDistance()
        self.hausdorff_calc = HausdorffDistance()
        self.p2s_calc = PointToSurfaceDistance()
    
    def add_quality_gate(self, name: str, metric_type: str,
                        threshold: float, operator: str = '<='):
        """
        Add a quality gate constraint.
        
        Args:
            name: Gate name (e.g., 'chamfer_distance')
            metric_type: Type of metric
            threshold: Threshold value
            operator: Comparison operator ('<', '<=', '>', '>=', '==')
        """
        gate = QualityGate(
            name=name,
            metric_type=metric_type,
            threshold=threshold,
            operator=operator
        )
        self.quality_gates[name] = gate
    
    def _check_gate(self, gate: QualityGate, value: float) -> bool:
        """
        Check if value passes quality gate.
        
        Args:
            gate: Quality gate
            value: Value to check
            
        Returns:
            True if passes, False otherwise
        """
        if gate.operator == '<':
            return value < gate.threshold
        elif gate.operator == '<=':
            return value <= gate.threshold
        elif gate.operator == '>':
            return value > gate.threshold
        elif gate.operator == '>=':
            return value >= gate.threshold
        elif gate.operator == '==':
            return abs(value - gate.threshold) < 1e-6
        return False
    
    def compute_metrics(self, source_path: str,
                       target_path: str) -> Dict[str, float]:
        """
        Compute all metrics between source and target.
        
        Args:
            source_path: Path to source file (.ply or .obj)
            target_path: Path to target file (.ply or .obj)
            
        Returns:
            Dictionary of computed metrics
        """
        metrics = {}
        
        try:
            # Load files
            if source_path.endswith(('.ply', '.pcd')):
                source = o3d.io.read_point_cloud(source_path)
            else:
                source_mesh = trimesh.load(source_path)
                source = o3d.geometry.PointCloud(
                    o3d.utility.Vector3dVector(source_mesh.vertices)
                )
            
            if target_path.endswith(('.ply', '.pcd')):
                target = o3d.io.read_point_cloud(target_path)
            else:
                target_mesh = trimesh.load(target_path)
                target = o3d.geometry.PointCloud(
                    o3d.utility.Vector3dVector(target_mesh.vertices)
                )
            
            # Compute metrics
            metrics['chamfer_distance'] = self.chamfer_calc.compute(source, target)
            metrics['hausdorff_distance'] = self.hausdorff_calc.compute(source, target)
            
            # Try to compute mesh-based metrics if both are meshes
            if not source_path.endswith(('.ply', '.pcd')) and \
               not target_path.endswith(('.ply', '.pcd')):
                try:
                    source_mesh = trimesh.load(source_path)
                    target_mesh = trimesh.load(target_path)
                    
                    if source_mesh.is_watertight and target_mesh.is_watertight:
                        metrics['volume_error'] = abs(source_mesh.volume - target_mesh.volume)
                    
                    metrics['surface_area_error'] = abs(source_mesh.area - target_mesh.area)
                except:
                    pass
        
        except Exception as e:
            metrics['error'] = str(e)
        
        return metrics
    
    def run(self, test_name: str, source_file: str,
            target_file: str) -> RegressionTestResult:
        """
        Run a single regression test.
        
        Args:
            test_name: Name of the test
            source_file: Path to source/predicted file
            target_file: Path to target/golden file
            
        Returns:
            RegressionTestResult with metrics and gate status
        """
        try:
            # Compute metrics
            metrics = self.compute_metrics(source_file, target_file)
            
            # Check quality gates
            quality_gates_status = {}
            all_passed = True
            
            for gate_name, gate in self.quality_gates.items():
                if gate.metric_type in metrics:
                    passed = self._check_gate(gate, metrics[gate.metric_type])
                    quality_gates_status[gate_name] = passed
                    if not passed:
                        all_passed = False
            
            result = RegressionTestResult(
                test_name=test_name,
                passed=all_passed,
                metrics=metrics,
                quality_gates=quality_gates_status
            )
        
        except Exception as e:
            result = RegressionTestResult(
                test_name=test_name,
                passed=False,
                metrics={},
                quality_gates={},
                error_message=str(e)
            )
        
        # Save result
        self._save_result(result)
        
        return result
    
    def run_batch(self, test_cases: List[tuple]) -> List[RegressionTestResult]:
        """
        Run multiple regression tests.
        
        Args:
            test_cases: List of (test_name, source_file, target_file) tuples
            
        Returns:
            List of TestResults
        """
        results = []
        for test_name, source_file, target_file in test_cases:
            result = self.run(test_name, source_file, target_file)
            results.append(result)
        
        return results
    
    def _save_result(self, result: RegressionTestResult):
        """Save test result to file."""
        result_file = self.output_dir / f"{result.test_name}_result.json"
        
        result_dict = asdict(result)
        with open(result_file, 'w') as f:
            json.dump(result_dict, f, indent=2)
    
    def generate_summary(self, results: List[RegressionTestResult]) -> Dict:
        """
        Generate summary of multiple test results.
        
        Args:
            results: List of TestResults
            
        Returns:
            Summary dictionary
        """
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.passed)
        failed_tests = total_tests - passed_tests
        
        # Aggregate metrics
        all_metrics = {}
        for result in results:
            for metric_name, value in result.metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                if isinstance(value, (int, float)):
                    all_metrics[metric_name].append(value)
        
        metric_stats = {}
        for metric_name, values in all_metrics.items():
            if values:
                metric_stats[metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                }
        
        summary = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'pass_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'metric_statistics': metric_stats,
            'test_results': [asdict(r) for r in results]
        }
        
        return summary
    
    def save_summary(self, results: List[RegressionTestResult],
                    filename: str = "test_summary.json"):
        """
        Save test summary to file.
        
        Args:
            results: List of TestResults
            filename: Output filename
        """
        summary = self.generate_summary(results)
        
        summary_file = self.output_dir / filename
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary_file
