import numpy as np
import matplotlib.pyplot as plt
import time
import psutil
import gc
import tracemalloc
import cProfile
import pstats
from io import StringIO
from scipy.integrate import solve_ivp
import seaborn as sns
from typing import Dict, List, Tuple, Any
import warnings

warnings.filterwarnings("ignore")


class HamiltonianBenchmarkSuite:
    """
    Comprehensive benchmarking suite for Hamiltonian analysis systems.

    Features:
    - Performance profiling (CPU, memory, I/O)
    - Computational complexity analysis
    - Optimization recommendations
    - Scalability testing
    - Memory leak detection
    - Algorithm efficiency comparison
    """

    def __init__(self):
        self.benchmark_results = {}
        self.performance_metrics = {}
        self.memory_usage = {}
        self.complexity_analysis = {}

    def softplus(self, x):
        return np.log(1 + np.exp(x))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def benchmark_hamiltonian_system(self, t_span=(0, 10), n_points=1000, params=None):
        """Benchmark the core Hamiltonian system computation."""
        print("🚀 Benchmarking Hamiltonian System Performance")
        print("=" * 50)

        if params is None:
            params = [0.3, 0.1, 0.2, 0.5, 0.5, 0.1]

        def hamiltonian_system(t, state, params):
            H, dH_dt = state
            eta, sigma, alpha1, tau, gamma, delta = params

            # Oscillatory terms
            A1 = 1.0 + 0.1 * np.sin(0.5 * t)
            A2 = 0.8 + 0.2 * np.cos(0.3 * t)
            A3 = 1.2 + 0.15 * np.sin(0.7 * t)

            B1 = 2.0 + 0.1 * t
            B2 = 1.5 + 0.05 * t
            B3 = 2.5 + 0.15 * t

            oscillatory = (
                A1 * np.sin(B1 * t)
                + A2 * np.sin(B2 * t + np.pi / 3)
                + A3 * np.sin(B3 * t + np.pi / 6)
            )

            # Drift terms
            drift = 0.01 * t**2 + alpha1 * np.sin(2 * np.pi * t) + 0.05 * np.log(1 + t)

            # Delayed feedback
            delayed_feedback = eta * H * self.sigmoid(gamma * H)

            # Stochastic term
            noise = np.random.normal(0, sigma * (1 + 0.2 * abs(H)))

            # External input
            external_input = delta * np.sin(3 * t) * np.exp(-0.2 * t)

            H_total = oscillatory + drift + delayed_feedback + noise + external_input

            return [dH_dt, H_total - H]

        # Memory tracking
        tracemalloc.start()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # CPU profiling
        profiler = cProfile.Profile()
        profiler.enable()

        # Timing
        start_time = time.time()

        # Run simulation
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        solution = solve_ivp(
            lambda t, state: hamiltonian_system(t, state, params),
            t_span,
            [0.0, 0.0],
            t_eval=t_eval,
            method="RK45",
            rtol=1e-8,
            atol=1e-10,
        )

        end_time = time.time()
        execution_time = end_time - start_time

        # Memory usage
        current, peak = tracemalloc.get_traced_memory()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = final_memory - initial_memory

        # CPU profiling results
        profiler.disable()
        s = StringIO()
        stats = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
        stats.print_stats(10)  # Top 10 functions

        # Performance metrics
        performance_metrics = {
            "execution_time": execution_time,
            "memory_used_mb": memory_used,
            "peak_memory_mb": peak / 1024 / 1024,
            "points_per_second": n_points / execution_time,
            "memory_per_point": memory_used / n_points,
            "cpu_profile": s.getvalue(),
        }

        tracemalloc.stop()

        print(f"⏱️  Execution Time: {execution_time:.4f} seconds")
        print(f"💾 Memory Used: {memory_used:.2f} MB")
        print(f"📊 Points per Second: {performance_metrics['points_per_second']:.0f}")
        print(f"🔍 Memory per Point: {performance_metrics['memory_per_point']:.4f} MB")

        self.performance_metrics["hamiltonian_system"] = performance_metrics
        return performance_metrics

    def benchmark_complexity_scaling(self, max_points=5000, step=500):
        """Benchmark computational complexity scaling."""
        print("\n📈 Complexity Scaling Analysis")
        print("=" * 40)

        point_counts = list(range(step, max_points + 1, step))
        execution_times = []
        memory_usage = []

        for n_points in point_counts:
            print(f"Testing {n_points} points...")

            # Memory tracking
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024

            start_time = time.time()

            # Run simulation
            t_eval = np.linspace(0, 10, n_points)
            solution = solve_ivp(
                lambda t, state: self._simple_hamiltonian(t, state),
                (0, 10),
                [0.0, 0.0],
                t_eval=t_eval,
                method="RK45",
            )

            end_time = time.time()
            final_memory = process.memory_info().rss / 1024 / 1024

            execution_times.append(end_time - start_time)
            memory_usage.append(final_memory - initial_memory)

        # Complexity analysis
        point_counts_array = np.array(point_counts)
        execution_times_array = np.array(execution_times)

        # Fit different complexity models
        models = {
            "O(n)": point_counts_array,
            "O(n log n)": point_counts_array * np.log(point_counts_array),
            "O(n²)": point_counts_array**2,
            "O(n³)": point_counts_array**3,
        }

        best_fit = None
        best_r_squared = 0

        for model_name, model_data in models.items():
            # Normalize for comparison
            normalized_model = model_data / model_data[0]
            normalized_times = execution_times_array / execution_times_array[0]

            # Calculate R-squared
            correlation = np.corrcoef(normalized_model, normalized_times)[0, 1]
            r_squared = correlation**2

            if r_squared > best_r_squared:
                best_r_squared = r_squared
                best_fit = model_name

        complexity_analysis = {
            "point_counts": point_counts,
            "execution_times": execution_times,
            "memory_usage": memory_usage,
            "best_fit_complexity": best_fit,
            "r_squared": best_r_squared,
        }

        print(f"🎯 Best Fit Complexity: {best_fit}")
        print(f"📊 R-squared: {best_r_squared:.4f}")

        self.complexity_analysis = complexity_analysis
        return complexity_analysis

    def _simple_hamiltonian(self, t, state):
        """Simplified Hamiltonian for complexity testing."""
        H, dH_dt = state
        return [dH_dt, np.sin(t) + 0.1 * H]

    def benchmark_memory_efficiency(self, n_runs=5):
        """Benchmark memory efficiency and detect leaks."""
        print("\n🔍 Memory Efficiency Analysis")
        print("=" * 40)

        memory_traces = []

        for i in range(n_runs):
            # Force garbage collection
            gc.collect()

            # Memory tracking
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024

            # Run simulation
            t_eval = np.linspace(0, 10, 1000)
            solution = solve_ivp(
                lambda t, state: self._simple_hamiltonian(t, state),
                (0, 10),
                [0.0, 0.0],
                t_eval=t_eval,
                method="RK45",
            )

            final_memory = process.memory_info().rss / 1024 / 1024
            memory_traces.append(final_memory - initial_memory)

            # Clear variables
            del solution
            gc.collect()

        # Memory leak detection
        memory_array = np.array(memory_traces)
        memory_trend = np.polyfit(range(len(memory_array)), memory_array, 1)[0]

        memory_analysis = {
            "memory_traces": memory_traces,
            "average_memory": np.mean(memory_traces),
            "memory_std": np.std(memory_traces),
            "memory_trend": memory_trend,
            "potential_leak": memory_trend > 0.1,  # MB per run
        }

        print(f"💾 Average Memory Usage: {memory_analysis['average_memory']:.2f} MB")
        print(f"📊 Memory Standard Deviation: {memory_analysis['memory_std']:.2f} MB")
        print(f"📈 Memory Trend: {memory_analysis['memory_trend']:.4f} MB/run")

        if memory_analysis["potential_leak"]:
            print("⚠️  Potential memory leak detected!")
        else:
            print("✅ No significant memory leak detected")

        self.memory_usage = memory_analysis
        return memory_analysis

    def benchmark_algorithm_comparison(self):
        """Compare different numerical integration methods."""
        print("\n⚡ Algorithm Comparison Benchmark")
        print("=" * 40)

        methods = ["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"]
        comparison_results = {}

        for method in methods:
            print(f"Testing {method}...")

            start_time = time.time()

            try:
                solution = solve_ivp(
                    lambda t, state: self._complex_hamiltonian(t, state),
                    (0, 10),
                    [0.0, 0.0],
                    t_eval=np.linspace(0, 10, 1000),
                    method=method,
                    rtol=1e-8,
                    atol=1e-10,
                )

                end_time = time.time()
                execution_time = end_time - start_time

                # Calculate accuracy (using RK45 as reference)
                if method == "RK45":
                    reference_solution = solution
                    accuracy = 1.0
                else:
                    # Compare with reference
                    max_diff = np.max(np.abs(solution.y - reference_solution.y))
                    accuracy = 1.0 / (1.0 + max_diff)

                comparison_results[method] = {
                    "execution_time": execution_time,
                    "accuracy": accuracy,
                    "success": True,
                }

            except Exception as e:
                comparison_results[method] = {
                    "execution_time": float("inf"),
                    "accuracy": 0.0,
                    "success": False,
                    "error": str(e),
                }

        # Find best method
        successful_methods = {k: v for k, v in comparison_results.items() if v["success"]}

        if successful_methods:
            best_method = min(successful_methods.items(), key=lambda x: x[1]["execution_time"])

            print(f"🏆 Best Method: {best_method[0]}")
            print(f"⏱️  Execution Time: {best_method[1]['execution_time']:.4f}s")
            print(f"🎯 Accuracy: {best_method[1]['accuracy']:.4f}")

        self.benchmark_results["algorithm_comparison"] = comparison_results
        return comparison_results

    def _complex_hamiltonian(self, t, state):
        """Complex Hamiltonian for algorithm comparison."""
        H, dH_dt = state

        # More complex system
        oscillatory = np.sin(2 * t) + 0.5 * np.sin(3 * t)
        drift = 0.01 * t**2 + 0.1 * np.sin(2 * np.pi * t)
        feedback = 0.3 * H * self.sigmoid(0.5 * H)
        noise = 0.1 * np.random.normal(0, 1)
        external = 0.1 * np.sin(3 * t) * np.exp(-0.2 * t)

        H_total = oscillatory + drift + feedback + noise + external

        return [dH_dt, H_total - H]

    def generate_benchmark_report(self):
        """Generate comprehensive benchmark report."""
        print("\n📊 Generating Benchmark Report...")

        report = f"""
# 🏆 Hamiltonian System Benchmark Report

## Performance Metrics

### Core System Performance
- Execution Time: {self.performance_metrics.get('hamiltonian_system', {}).get('execution_time', 'N/A'):.4f} seconds
- Memory Usage: {self.performance_metrics.get('hamiltonian_system', {}).get('memory_used_mb', 'N/A'):.2f} MB
- Points per Second: {self.performance_metrics.get('hamiltonian_system', {}).get('points_per_second', 'N/A'):.0f}
- Memory per Point: {self.performance_metrics.get('hamiltonian_system', {}).get('memory_per_point', 'N/A'):.4f} MB

### Complexity Analysis
- Best Fit Complexity: {self.complexity_analysis.get('best_fit_complexity', 'N/A')}
- R-squared: {self.complexity_analysis.get('r_squared', 'N/A'):.4f}

### Memory Efficiency
- Average Memory Usage: {self.memory_usage.get('average_memory', 'N/A'):.2f} MB
- Memory Standard Deviation: {self.memory_usage.get('memory_std', 'N/A'):.2f} MB
- Memory Trend: {self.memory_usage.get('memory_trend', 'N/A'):.4f} MB/run
- Memory Leak Detected: {'Yes' if self.memory_usage.get('potential_leak', False) else 'No'}

### Algorithm Comparison
"""

        if "algorithm_comparison" in self.benchmark_results:
            for method, results in self.benchmark_results["algorithm_comparison"].items():
                if results["success"]:
                    report += f"- {method}: {results['execution_time']:.4f}s, Accuracy: {results['accuracy']:.4f}\n"
                else:
                    report += f"- {method}: Failed ({results.get('error', 'Unknown error')})\n"

        report += f"""
## Optimization Recommendations

### Performance Optimizations
1. **Algorithm Selection**: Use the fastest successful method for your specific problem
2. **Memory Management**: Implement proper cleanup for large-scale simulations
3. **Parallelization**: Consider parallel processing for parameter sweeps
4. **Caching**: Cache frequently computed values

### Scalability Considerations
1. **Complexity**: The system shows {self.complexity_analysis.get('best_fit_complexity', 'unknown')} scaling
2. **Memory**: Monitor memory usage for large-scale simulations
3. **Accuracy**: Balance speed vs. accuracy based on requirements

## System Requirements
- Minimum RAM: {max(self.memory_usage.get('average_memory', 0) * 2, 512):.0f} MB
- Recommended RAM: {max(self.memory_usage.get('average_memory', 0) * 4, 1024):.0f} MB
- CPU: Multi-core recommended for parallel processing

---
*Generated by Hamiltonian Benchmark Suite*
"""

        with open("hamiltonian_benchmark_report.md", "w") as f:
            f.write(report)

        print("📄 Benchmark report saved as 'hamiltonian_benchmark_report.md'")
        return report

    def create_benchmark_visualizations(self):
        """Create comprehensive benchmark visualizations."""
        print("\n📈 Creating Benchmark Visualizations...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Complexity scaling
        if self.complexity_analysis:
            ax1 = axes[0, 0]
            ax1.plot(
                self.complexity_analysis["point_counts"],
                self.complexity_analysis["execution_times"],
                "bo-",
            )
            ax1.set_xlabel("Number of Points")
            ax1.set_ylabel("Execution Time (s)")
            ax1.set_title("Complexity Scaling")
            ax1.grid(True, alpha=0.3)

        # 2. Memory usage over time
        if self.memory_usage:
            ax2 = axes[0, 1]
            ax2.plot(self.memory_usage["memory_traces"], "ro-")
            ax2.set_xlabel("Run Number")
            ax2.set_ylabel("Memory Usage (MB)")
            ax2.set_title("Memory Efficiency")
            ax2.grid(True, alpha=0.3)

        # 3. Algorithm comparison
        if "algorithm_comparison" in self.benchmark_results:
            ax3 = axes[0, 2]
            methods = []
            times = []
            for method, results in self.benchmark_results["algorithm_comparison"].items():
                if results["success"]:
                    methods.append(method)
                    times.append(results["execution_time"])

            if methods:
                bars = ax3.bar(methods, times)
                ax3.set_ylabel("Execution Time (s)")
                ax3.set_title("Algorithm Performance")
                ax3.tick_params(axis="x", rotation=45)

        # 4. Performance metrics
        if self.performance_metrics.get("hamiltonian_system"):
            ax4 = axes[1, 0]
            metrics = self.performance_metrics["hamiltonian_system"]
            labels = ["Time (s)", "Memory (MB)", "Points/s"]
            values = [
                metrics["execution_time"],
                metrics["memory_used_mb"],
                metrics["points_per_second"] / 1000,
            ]  # Scale for visualization

            bars = ax4.bar(labels, values)
            ax4.set_title("Performance Metrics")
            ax4.tick_params(axis="x", rotation=45)

        # 5. Memory trend analysis
        if self.memory_usage:
            ax5 = axes[1, 1]
            x = np.arange(len(self.memory_usage["memory_traces"]))
            ax5.scatter(x, self.memory_usage["memory_traces"], alpha=0.6)

            # Trend line
            z = np.polyfit(x, self.memory_usage["memory_traces"], 1)
            p = np.poly1d(z)
            ax5.plot(x, p(x), "r--", alpha=0.8)

            ax5.set_xlabel("Run Number")
            ax5.set_ylabel("Memory Usage (MB)")
            ax5.set_title("Memory Trend Analysis")
            ax5.grid(True, alpha=0.3)

        # 6. Summary statistics
        ax6 = axes[1, 2]
        ax6.axis("off")

        summary_text = f"""
Benchmark Summary

Performance:
• Execution Time: {self.performance_metrics.get('hamiltonian_system', {}).get('execution_time', 'N/A'):.4f}s
• Memory Usage: {self.performance_metrics.get('hamiltonian_system', {}).get('memory_used_mb', 'N/A'):.2f}MB
• Points/sec: {self.performance_metrics.get('hamiltonian_system', {}).get('points_per_second', 'N/A'):.0f}

Complexity:
• Best Fit: {self.complexity_analysis.get('best_fit_complexity', 'N/A')}
• R²: {self.complexity_analysis.get('r_squared', 'N/A'):.4f}

Memory:
• Average: {self.memory_usage.get('average_memory', 'N/A'):.2f}MB
• Leak: {'Yes' if self.memory_usage.get('potential_leak', False) else 'No'}
        """

        ax6.text(
            0.1,
            0.9,
            summary_text,
            transform=ax6.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        )

        plt.tight_layout()
        plt.savefig("hamiltonian_benchmark_visualizations.png", dpi=300, bbox_inches="tight")
        plt.show()

        print("📊 Benchmark visualizations saved as 'hamiltonian_benchmark_visualizations.png'")


def run_comprehensive_benchmark():
    """Run the complete benchmark suite."""
    print("🏆 Comprehensive Hamiltonian Benchmark Suite")
    print("=" * 60)

    benchmark = HamiltonianBenchmarkSuite()

    # 1. Core system performance
    performance = benchmark.benchmark_hamiltonian_system()

    # 2. Complexity scaling
    complexity = benchmark.benchmark_complexity_scaling()

    # 3. Memory efficiency
    memory = benchmark.benchmark_memory_efficiency()

    # 4. Algorithm comparison
    algorithms = benchmark.benchmark_algorithm_comparison()

    # 5. Generate report
    report = benchmark.generate_benchmark_report()

    # 6. Create visualizations
    benchmark.create_benchmark_visualizations()

    print("\n🏆 Benchmark Suite Complete!")
    print("Files generated:")
    print("  - hamiltonian_benchmark_report.md")
    print("  - hamiltonian_benchmark_visualizations.png")

    return benchmark


if __name__ == "__main__":
    # Run the comprehensive benchmark
    benchmark = run_comprehensive_benchmark()
