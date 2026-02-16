"""
Auto-generated comparison reports with visual overlays and metrics.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
from jinja2 import Template

from .metrics import ChamferDistance, PointToSurfaceDistance, HausdorffDistance
from .visualization import ComparisonOverlay


class ComparisonReporter:
    """
    Generate HTML and JSON reports comparing meshes/point clouds.
    """
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize reporter.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(self, source_file: str,
                       target_file: str,
                       test_name: str = "Comparison Report",
                       metrics: Optional[Dict] = None,
                       alignment_result: Optional[Dict] = None,
                       quality_gates: Optional[Dict] = None) -> Dict:
        """
        Generate HTML and JSON reports.
        
        Args:
            source_file: Path to source file
            target_file: Path to target file
            test_name: Name of the test/comparison
            metrics: Dictionary of computed metrics
            alignment_result: ICP alignment result data
            quality_gates: Quality gate check results
            
        Returns:
            Report metadata dictionary
        """
        report_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Build report data
        report_data = {
            'report_id': report_id,
            'timestamp': datetime.now().isoformat(),
            'test_name': test_name,
            'source_file': str(source_file),
            'target_file': str(target_file),
            'metrics': metrics or {},
            'alignment': alignment_result or {},
            'quality_gates': quality_gates or {},
        }
        
        # Generate JSON report
        json_path = self._generate_json_report(report_data)
        
        # Generate HTML report
        html_path = self._generate_html_report(report_data)
        
        report_data['json_path'] = str(json_path)
        report_data['html_path'] = str(html_path)
        
        return report_data
    
    def _generate_json_report(self, report_data: Dict) -> Path:
        """Generate JSON report file."""
        report_id = report_data['report_id']
        json_file = self.output_dir / f"report_{report_id}.json"
        
        with open(json_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        return json_file
    
    def _generate_html_report(self, report_data: Dict) -> Path:
        """Generate HTML report file."""
        report_id = report_data['report_id']
        html_file = self.output_dir / f"report_{report_id}.html"
        
        html_content = self._create_html_content(report_data)
        
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        return html_file
    
    def _create_html_content(self, report_data: Dict) -> str:
        """Create HTML content for report."""
        
        # Format metrics table
        metrics_rows = ""
        for metric_name, metric_value in report_data['metrics'].items():
            if isinstance(metric_value, float):
                metrics_rows += f"""
                <tr>
                    <td>{metric_name}</td>
                    <td>{metric_value:.6f}</td>
                </tr>
                """
            else:
                metrics_rows += f"""
                <tr>
                    <td>{metric_name}</td>
                    <td>{metric_value}</td>
                </tr>
                """
        
        # Format quality gates
        quality_gates_rows = ""
        for gate_name, gate_passed in report_data['quality_gates'].items():
            status = "✓ PASS" if gate_passed else "✗ FAIL"
            status_class = "pass" if gate_passed else "fail"
            quality_gates_rows += f"""
            <tr>
                <td>{gate_name}</td>
                <td class="{status_class}">{status}</td>
            </tr>
            """
        
        # Format alignment info
        alignment_info = ""
        if report_data['alignment']:
            alignment = report_data['alignment']
            alignment_info = f"""
            <div class="section">
                <h3>Alignment Results</h3>
                <table class="metrics-table">
                    <tr><td>Fitness</td><td>{alignment.get('fitness', 'N/A')}</td></tr>
                    <tr><td>RMSE</td><td>{alignment.get('rmse', 'N/A')}</td></tr>
                    <tr><td>Converged</td><td>{alignment.get('converged', 'N/A')}</td></tr>
                </table>
            </div>
            """
        
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>3D Validation Report - {test_name}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1000px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #333;
                    border-bottom: 3px solid #0066cc;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #555;
                    margin-top: 30px;
                }}
                h3 {{
                    color: #666;
                }}
                .section {{
                    margin: 20px 0;
                    padding: 15px;
                    background-color: #fafafa;
                    border-left: 4px solid #0066cc;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 10px 0;
                }}
                table.metrics-table {{
                    width: 100%;
                }}
                th {{
                    background-color: #0066cc;
                    color: white;
                    padding: 12px;
                    text-align: left;
                }}
                td {{
                    padding: 10px;
                    border-bottom: 1px solid #ddd;
                }}
                tr:hover {{
                    background-color: #f0f0f0;
                }}
                .pass {{
                    color: green;
                    font-weight: bold;
                }}
                .fail {{
                    color: red;
                    font-weight: bold;
                }}
                .meta {{
                    color: #888;
                    font-size: 0.9em;
                }}
                .file-path {{
                    font-family: monospace;
                    background-color: #f0f0f0;
                    padding: 5px;
                    border-radius: 3px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>3D Validation Report</h1>
                <p class="meta">Report ID: {report_id} | Generated: {timestamp}</p>
                
                <h2>{test_name}</h2>
                
                <div class="section">
                    <h3>Files Compared</h3>
                    <p><strong>Source:</strong> <span class="file-path">{source_file}</span></p>
                    <p><strong>Target:</strong> <span class="file-path">{target_file}</span></p>
                </div>
                
                <div class="section">
                    <h3>Metrics</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Metric</th>
                                <th>Value</th>
                            </tr>
                        </thead>
                        <tbody>
                            {metrics_rows}
                        </tbody>
                    </table>
                </div>
                
                {alignment_info}
                
                <div class="section">
                    <h3>Quality Gates</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Gate</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            {quality_gates_rows}
                        </tbody>
                    </table>
                </div>
                
                <div class="section">
                    <h3>Summary</h3>
                    <p><strong>Overall Status:</strong> 
                        <span class="{'pass' if all(report_data['quality_gates'].values()) and report_data['quality_gates'] else 'fail'}">
                            {'✓ PASS' if all(report_data['quality_gates'].values()) and report_data['quality_gates'] else '✗ FAIL'}
                        </span>
                    </p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_template.format(
            report_id=report_data['report_id'],
            timestamp=report_data['timestamp'],
            test_name=report_data['test_name'],
            source_file=report_data['source_file'],
            target_file=report_data['target_file'],
            metrics_rows=metrics_rows,
            alignment_info=alignment_info,
            quality_gates_rows=quality_gates_rows
        )
    
    def generate_comparison_summary(self, reports: List[Dict]) -> str:
        """
        Generate a summary HTML comparing multiple reports.
        
        Args:
            reports: List of report dictionaries
            
        Returns:
            Path to summary HTML file
        """
        summary_file = self.output_dir / "comparison_summary.html"
        
        # Build comparison table
        comparison_rows = ""
        for report in reports:
            test_name = report.get('test_name', 'Unknown')
            source = report.get('source_file', 'N/A').split('/')[-1]
            target = report.get('target_file', 'N/A').split('/')[-1]
            
            # Get status
            quality_gates = report.get('quality_gates', {})
            status = "✓ PASS" if all(quality_gates.values()) else "✗ FAIL" if quality_gates else "N/A"
            status_class = "pass" if (status == "✓ PASS") else "fail" if (status == "✗ FAIL") else "neutral"
            
            # Get key metric
            metrics = report.get('metrics', {})
            chamfer = metrics.get('chamfer_distance', 'N/A')
            
            comparison_rows += f"""
            <tr>
                <td>{test_name}</td>
                <td>{source}</td>
                <td>{target}</td>
                <td class="{status_class}">{status}</td>
                <td>{chamfer if isinstance(chamfer, str) else f'{chamfer:.6f}'}</td>
            </tr>
            """
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Comparison Summary</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #333;
                    border-bottom: 3px solid #0066cc;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th {{
                    background-color: #0066cc;
                    color: white;
                    padding: 12px;
                    text-align: left;
                }}
                td {{
                    padding: 10px;
                    border-bottom: 1px solid #ddd;
                }}
                tr:hover {{
                    background-color: #f0f0f0;
                }}
                .pass {{
                    color: green;
                    font-weight: bold;
                }}
                .fail {{
                    color: red;
                    font-weight: bold;
                }}
                .neutral {{
                    color: #999;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Validation Comparison Summary</h1>
                <p>Generated: {datetime.now().isoformat()}</p>
                
                <table>
                    <thead>
                        <tr>
                            <th>Test Name</th>
                            <th>Source</th>
                            <th>Target</th>
                            <th>Status</th>
                            <th>Chamfer Distance</th>
                        </tr>
                    </thead>
                    <tbody>
                        {comparison_rows}
                    </tbody>
                </table>
            </div>
        </body>
        </html>
        """
        
        with open(summary_file, 'w') as f:
            f.write(html_content)
        
        return str(summary_file)
