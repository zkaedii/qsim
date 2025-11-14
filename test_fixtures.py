#!/usr/bin/env python3
"""
Test Fixtures for Claude Analysis Agent V2

Provides mock data, test utilities, and fixtures for local testing
without requiring actual performance data files.
"""

import json
from typing import Dict, Any, List
from datetime import datetime, timedelta


class TestDataGenerator:
    """Generate realistic test data for analysis."""
    
    @staticmethod
    def generate_performance_comparison() -> Dict[str, Any]:
        """Generate mock performance comparison data."""
        return {
            'performance_evolution': {
                'jit_compiled_v1': 75000000,
                'jit_compiled_v2': 85000000,
                'jit_compiled_v3': 95000000,
                'vectorized_numpy': 45000000,
                'async_concurrent': 30000000,
                'thread_pool_optimized': 55000000,
                'process_pool': 25000000,
                'basic_sequential': 5000000,
                'memory_optimized': 65000000,
                'hybrid_approach': 70000000
            },
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0',
                'test_environment': 'staging'
            }
        }
    
    @staticmethod
    def generate_ultimate_speed_results() -> Dict[str, Any]:
        """Generate mock ultimate speed optimization results."""
        return {
            'benchmark_results': {
                'ultra_fast_jit': {
                    'tasks_per_second': 95000000,
                    'processing_time_per_task_ns': 10.5,
                    'total_tasks': 1000000,
                    'duration_seconds': 10.5
                },
                'vectorized_simd': {
                    'tasks_per_second': 75000000,
                    'processing_time_per_task_ns': 13.3,
                    'total_tasks': 1000000,
                    'duration_seconds': 13.3
                },
                'parallel_threaded': {
                    'tasks_per_second': 55000000,
                    'processing_time_per_task_ns': 18.2,
                    'total_tasks': 1000000,
                    'duration_seconds': 18.2
                },
                'async_io_bound': {
                    'tasks_per_second': 40000000,
                    'processing_time_per_task_ns': 25.0,
                    'total_tasks': 1000000,
                    'duration_seconds': 25.0
                },
                'standard_sequential': {
                    'tasks_per_second': 8000000,
                    'processing_time_per_task_ns': 125.0,
                    'total_tasks': 1000000,
                    'duration_seconds': 125.0
                }
            },
            'optimization_summary': {
                'best_method': 'ultra_fast_jit',
                'improvement_factor': 11.875,
                'total_methods_tested': 5
            }
        }
    
    @staticmethod
    def generate_parallel_scaling_results() -> Dict[str, Any]:
        """Generate mock parallel scaling performance results."""
        return {
            'benchmark_results': {
                '100_tasks_thread_pool': {
                    'tasks_per_second': 25000000,
                    'processing_time_per_task_ns': 40.0,
                    'total_tasks': 100,
                    'duration_seconds': 0.004
                },
                '1000_tasks_thread_pool': {
                    'tasks_per_second': 45000000,
                    'processing_time_per_task_ns': 22.2,
                    'total_tasks': 1000,
                    'duration_seconds': 0.022
                },
                '10000_tasks_thread_pool': {
                    'tasks_per_second': 65000000,
                    'processing_time_per_task_ns': 15.4,
                    'total_tasks': 10000,
                    'duration_seconds': 0.154
                },
                '100000_tasks_thread_pool': {
                    'tasks_per_second': 75000000,
                    'processing_time_per_task_ns': 13.3,
                    'total_tasks': 100000,
                    'duration_seconds': 1.33
                }
            },
            'scaling_analysis': {
                'scaling_type': 'sub_linear',
                'efficiency': 0.75,
                'optimal_batch_size': 100000
            }
        }
    
    @staticmethod
    def generate_optimization_history() -> Dict[str, Any]:
        """Generate mock optimization history."""
        history = []
        base_time = datetime.now()
        
        for i in range(5):
            history.append({
                'timestamp': (base_time - timedelta(days=i)).isoformat(),
                'version': f'v1.{5-i}',
                'performance_rps': 50000000 + (i * 10000000),
                'optimization_applied': [
                    'JIT compilation',
                    'Vectorization',
                    'Memory pooling',
                    'Cache optimization'
                ][i] if i < 4 else 'All optimizations',
                'improvement_pct': 20.0 * (i + 1)
            })
        
        return {
            'optimization_timeline': history,
            'total_improvements': len(history),
            'cumulative_improvement': 150.0
        }
    
    @staticmethod
    def generate_invalid_data() -> Dict[str, Any]:
        """Generate invalid data for testing error handling."""
        return {
            'performance_evolution': {
                'bad_method_1': -1000,  # Negative RPS
                'bad_method_2': float('inf'),  # Infinite value
                'bad_method_3': None,  # None value
            }
        }
    
    @staticmethod
    def generate_insufficient_data() -> Dict[str, Any]:
        """Generate insufficient data for testing minimum requirements."""
        return {
            'performance_evolution': {
                'single_method': 5000000
            }
        }
    
    @staticmethod
    def save_test_data_files(directory: str = "."):
        """
        Save all test data to files for testing.
        
        Args:
            directory: Directory to save files in
        """
        files = {
            'complete_performance_comparison.json': TestDataGenerator.generate_performance_comparison(),
            'ultimate_speed_optimization_results.json': TestDataGenerator.generate_ultimate_speed_results(),
            'parallel_scaling_performance_results.json': TestDataGenerator.generate_parallel_scaling_results(),
            'claude_optimization_history.json': TestDataGenerator.generate_optimization_history()
        }
        
        for filename, data in files.items():
            filepath = f"{directory}/{filename}"
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        
        print(f"✅ Test data files created in {directory}/")
        return list(files.keys())


class MockDataSource:
    """Mock data source for testing without file I/O."""
    
    def __init__(self, data: Dict[str, Any]):
        """
        Initialize mock data source.
        
        Args:
            data: Mock data to return
        """
        self.data = data
        self.access_count = 0
    
    def load(self) -> Dict[str, Any]:
        """Load mock data."""
        self.access_count += 1
        return self.data
    
    def reset_counter(self):
        """Reset access counter."""
        self.access_count = 0


class TestScenarios:
    """Pre-defined test scenarios for various use cases."""
    
    @staticmethod
    def normal_operation() -> Dict[str, MockDataSource]:
        """Test scenario: Normal operation with valid data."""
        return {
            'performance_comparison': MockDataSource(
                TestDataGenerator.generate_performance_comparison()
            ),
            'ultimate_speed': MockDataSource(
                TestDataGenerator.generate_ultimate_speed_results()
            ),
            'parallel_scaling': MockDataSource(
                TestDataGenerator.generate_parallel_scaling_results()
            ),
            'optimization_history': MockDataSource(
                TestDataGenerator.generate_optimization_history()
            )
        }
    
    @staticmethod
    def partial_data() -> Dict[str, MockDataSource]:
        """Test scenario: Only some data sources available."""
        return {
            'performance_comparison': MockDataSource(
                TestDataGenerator.generate_performance_comparison()
            )
        }
    
    @staticmethod
    def no_data() -> Dict[str, MockDataSource]:
        """Test scenario: No data sources available."""
        return {}
    
    @staticmethod
    def invalid_data() -> Dict[str, MockDataSource]:
        """Test scenario: Invalid/corrupted data."""
        return {
            'performance_comparison': MockDataSource(
                TestDataGenerator.generate_invalid_data()
            )
        }
    
    @staticmethod
    def insufficient_data() -> Dict[str, MockDataSource]:
        """Test scenario: Insufficient data for analysis."""
        return {
            'performance_comparison': MockDataSource(
                TestDataGenerator.generate_insufficient_data()
            )
        }


class PerformanceAssertion:
    """Utilities for asserting performance characteristics in tests."""
    
    @staticmethod
    def assert_rps_within_range(
        actual_rps: float,
        expected_rps: float,
        tolerance_pct: float = 10.0
    ):
        """
        Assert RPS is within acceptable range.
        
        Args:
            actual_rps: Actual RPS value
            expected_rps: Expected RPS value
            tolerance_pct: Acceptable tolerance percentage
            
        Raises:
            AssertionError: If RPS is outside tolerance
        """
        lower_bound = expected_rps * (1 - tolerance_pct / 100)
        upper_bound = expected_rps * (1 + tolerance_pct / 100)
        
        assert lower_bound <= actual_rps <= upper_bound, (
            f"RPS {actual_rps:,.0f} outside expected range "
            f"[{lower_bound:,.0f}, {upper_bound:,.0f}]"
        )
    
    @staticmethod
    def assert_performance_improvement(
        baseline_rps: float,
        optimized_rps: float,
        min_improvement_pct: float = 10.0
    ):
        """
        Assert performance improvement meets minimum threshold.
        
        Args:
            baseline_rps: Baseline performance
            optimized_rps: Optimized performance
            min_improvement_pct: Minimum improvement percentage
            
        Raises:
            AssertionError: If improvement is insufficient
        """
        improvement_pct = ((optimized_rps - baseline_rps) / baseline_rps) * 100
        
        assert improvement_pct >= min_improvement_pct, (
            f"Performance improvement {improvement_pct:.1f}% "
            f"below minimum {min_improvement_pct}%"
        )
    
    @staticmethod
    def assert_statistical_significance(
        p_value: float,
        alpha: float = 0.05
    ):
        """
        Assert statistical significance.
        
        Args:
            p_value: P-value from statistical test
            alpha: Significance level
            
        Raises:
            AssertionError: If not statistically significant
        """
        assert p_value < alpha, (
            f"Result not statistically significant: "
            f"p={p_value:.4f} >= α={alpha}"
        )


def main():
    """Demonstrate test fixture usage."""
    print("🧪 CLAUDE ANALYSIS AGENT V2 - TEST FIXTURES")
    print("=" * 60)
    
    # Generate and save test data
    print("\n📁 Generating test data files...")
    files = TestDataGenerator.save_test_data_files()
    print(f"   Created {len(files)} test data files")
    
    # Demonstrate mock data source
    print("\n🔧 Mock data source example...")
    mock_source = MockDataSource(TestDataGenerator.generate_performance_comparison())
    data = mock_source.load()
    print(f"   Loaded data with {len(data['performance_evolution'])} methods")
    print(f"   Access count: {mock_source.access_count}")
    
    # Demonstrate test scenarios
    print("\n📋 Available test scenarios:")
    scenarios = [
        "normal_operation",
        "partial_data",
        "no_data",
        "invalid_data",
        "insufficient_data"
    ]
    for scenario in scenarios:
        print(f"   ✓ {scenario}")
    
    print("\n✅ Test fixtures ready for use!")


if __name__ == "__main__":
    main()
