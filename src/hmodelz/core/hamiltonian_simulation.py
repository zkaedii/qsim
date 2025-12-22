import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import expit
from typing import Dict, Tuple
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class ComplexHamiltonianSimulator:
    """
    Simulates the complex time-dependent Hamiltonian:

    H(t) = Σ[A_i(t)sin(B_i(t)t + φ_i) + C_i e^(-D_i t)]
           + ∫₀ᵗ softplus(a(x-x₀)² + b) f(x) g'(x) dx
           + α₀t² + α₁sin(2πt) + α₂log(1+t)
           + ηH(t-τ)σ(γH(t-τ)) + σN(0, 1+β|H(t-1)|) + δu(t)
    """

    def __init__(self, config: Dict = None):
        """Initialize the simulator with default or custom parameters."""
        self.default_config = {
            # Oscillatory terms (n=3 oscillators)
            "n_oscillators": 3,
            "A_functions": [
                # Time-varying amplitude
                lambda t: 1.0 + 0.1 * np.sin(0.5 * t),
                lambda t: 0.8 + 0.2 * np.cos(0.3 * t),
                lambda t: 1.2 + 0.15 * np.sin(0.7 * t),
            ],
            "B_functions": [
                lambda t: 2.0 + 0.1 * t,  # Time-varying frequency
                lambda t: 1.5 + 0.05 * t,
                lambda t: 2.5 + 0.15 * t,
            ],
            "C_values": [0.5, 0.3, 0.7],  # Decay amplitudes
            "D_values": [0.1, 0.2, 0.15],  # Decay rates
            "phi_values": [0, np.pi / 3, np.pi / 6],  # Phase shifts
            # Integral term parameters
            "a": 0.1,
            "b": 0.5,
            "x0": 1.0,
            # Drift parameters
            "alpha0": 0.01,  # Quadratic drift
            "alpha1": 0.2,  # Periodic amplitude
            "alpha2": 0.05,  # Logarithmic amplitude
            # Delayed feedback parameters
            "eta": 0.3,  # Feedback strength
            "gamma": 0.5,  # Nonlinear scaling
            "tau": 0.5,  # Delay time
            # Stochastic parameters
            "sigma": 0.1,  # Noise amplitude
            "beta": 0.2,  # State-dependent noise scaling
            # External input
            "delta": 0.1,  # Input scaling
            # Simulation parameters
            "dt": 0.01,  # Time step
            "t_max": 10.0,  # Maximum time
            "history_size": 1000,  # For delay calculations
        }

        self.config = {**self.default_config, **(config or {})}
        self.time_history = []
        self.H_history = []
        self.component_contributions = {
            "oscillatory": [],
            "integral": [],
            "drift": [],
            "delayed_feedback": [],
            "stochastic": [],
            "external_input": [],
        }

    def softplus(self, x: float) -> float:
        """Smooth, non-negative activation function."""
        return np.log(1 + np.exp(x))

    def sigmoid(self, x: float) -> float:
        """Sigmoid activation function using scipy.special.expit for numerical stability."""
        return expit(x)

    def f_function(self, x: float) -> float:
        """Arbitrary function f(x) for the integral term."""
        return np.sin(x) * np.exp(-0.1 * x)

    def g_prime_function(self, x: float) -> float:
        """Derivative of g(x) for the integral term."""
        return np.cos(x) * np.exp(-0.05 * x)

    def u_function(self, t: float) -> float:
        """External input function u(t)."""
        return np.sin(3 * t) * np.exp(-0.2 * t)

    def oscillatory_term(self, t: float) -> float:
        """Calculate the oscillatory and decaying sum term."""
        result = 0.0
        for i in range(self.config["n_oscillators"]):
            A_t = self.config["A_functions"][i](t)
            B_t = self.config["B_functions"][i](t)
            C_i = self.config["C_values"][i]
            D_i = self.config["D_values"][i]
            phi_i = self.config["phi_values"][i]

            oscillatory = A_t * np.sin(B_t * t + phi_i)
            decaying = C_i * np.exp(-D_i * t)
            result += oscillatory + decaying

        return result

    def integral_term(self, t: float) -> float:
        """Calculate the nonlinear integral term."""

        def integrand(x):
            softplus_arg = self.config["a"] * (x - self.config["x0"]) ** 2 + self.config["b"]
            return self.softplus(softplus_arg) * self.f_function(x) * self.g_prime_function(x)

        try:
            result, _ = quad(integrand, 0, t)
            return result
        except Exception:
            # Fallback for numerical issues
            return 0.0

    def drift_terms(self, t: float) -> float:
        """Calculate the polynomial, periodic, and logarithmic drift terms."""
        quadratic = self.config["alpha0"] * t**2
        periodic = self.config["alpha1"] * np.sin(2 * np.pi * t)
        logarithmic = self.config["alpha2"] * np.log(1 + t)

        return quadratic + periodic + logarithmic

    def delayed_feedback_term(self, t: float) -> float:
        """Calculate the delayed nonlinear feedback term."""
        if len(self.H_history) == 0:
            return 0.0

        # Find H(t-τ) using interpolation
        tau = self.config["tau"]
        target_time = t - tau

        if target_time <= 0:
            return 0.0

        # Simple linear interpolation for H(t-τ)
        if len(self.time_history) > 1:
            # Find the closest time points
            time_array = np.array(self.time_history)
            H_array = np.array(self.H_history)

            # Find indices for interpolation
            idx = np.searchsorted(time_array, target_time)
            if idx == 0:
                H_delayed = H_array[0]
            elif idx >= len(H_array):
                H_delayed = H_array[-1]
            else:
                # Linear interpolation
                t1, t2 = time_array[idx - 1], time_array[idx]
                H1, H2 = H_array[idx - 1], H_array[idx]
                H_delayed = H1 + (H2 - H1) * (target_time - t1) / (t2 - t1)
        else:
            H_delayed = self.H_history[-1] if self.H_history else 0.0

        return self.config["eta"] * H_delayed * self.sigmoid(self.config["gamma"] * H_delayed)

    def stochastic_term(self, t: float) -> float:
        """Calculate the state-dependent stochastic term."""
        if len(self.H_history) == 0:
            H_prev = 0.0
        else:
            # Get H(t-1) with interpolation
            target_time = t - 1.0
            if target_time <= 0:
                H_prev = 0.0
            else:
                time_array = np.array(self.time_history)
                H_array = np.array(self.H_history)
                idx = np.searchsorted(time_array, target_time)
                if idx == 0:
                    H_prev = H_array[0]
                elif idx >= len(H_array):
                    H_prev = H_array[-1]
                else:
                    t1, t2 = time_array[idx - 1], time_array[idx]
                    H1, H2 = H_array[idx - 1], H_array[idx]
                    H_prev = H1 + (H2 - H1) * (target_time - t1) / (t2 - t1)

        variance = 1 + self.config["beta"] * abs(H_prev)
        noise = np.random.normal(0, np.sqrt(variance))
        return self.config["sigma"] * noise

    def external_input_term(self, t: float) -> float:
        """Calculate the external input term."""
        return self.config["delta"] * self.u_function(t)

    def compute_H(self, t: float) -> float:
        """Compute the complete Hamiltonian H(t)."""
        # Compute each component
        oscillatory = self.oscillatory_term(t)
        integral = self.integral_term(t)
        drift = self.drift_terms(t)
        delayed_feedback = self.delayed_feedback_term(t)
        stochastic = self.stochastic_term(t)
        external_input = self.external_input_term(t)

        # Store component contributions
        self.component_contributions["oscillatory"].append(oscillatory)
        self.component_contributions["integral"].append(integral)
        self.component_contributions["drift"].append(drift)
        self.component_contributions["delayed_feedback"].append(delayed_feedback)
        self.component_contributions["stochastic"].append(stochastic)
        self.component_contributions["external_input"].append(external_input)

        # Combine all terms
        H_total = oscillatory + integral + drift + delayed_feedback + stochastic + external_input

        return H_total

    def simulate(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Run the complete simulation."""
        print("🚀 Starting Complex Hamiltonian Simulation...")
        print(f"📊 Time range: 0 to {self.config['t_max']} with dt = {self.config['dt']}")

        t_values = np.arange(0, self.config["t_max"], self.config["dt"])
        H_values = []

        for i, t in enumerate(t_values):
            if i % 100 == 0:
                print(f"⏱️  Progress: {i}/{len(t_values)} ({i/len(t_values)*100:.1f}%)")

            H = self.compute_H(t)
            H_values.append(H)

            # Store history for delay calculations
            self.time_history.append(t)
            self.H_history.append(H)

            # Keep history size manageable
            if len(self.time_history) > self.config["history_size"]:
                self.time_history = self.time_history[-self.config["history_size"] :]
                self.H_history = self.H_history[-self.config["history_size"] :]

        print("✅ Simulation complete!")
        return t_values, np.array(H_values), self.component_contributions

    def analyze_behavior(self, t_values: np.ndarray, H_values: np.ndarray, components: Dict):
        """Analyze the emergent behavior and interactions."""
        print("\n🔍 Analyzing Emergent Behavior...")

        # Convert component lists to arrays
        component_arrays = {k: np.array(v) for k, v in components.items()}

        # Calculate statistics
        stats = {
            "mean": np.mean(H_values),
            "std": np.std(H_values),
            "min": np.min(H_values),
            "max": np.max(H_values),
            "range": np.max(H_values) - np.min(H_values),
            "variance": np.var(H_values),
        }

        # Calculate correlations between components
        correlations = {}
        for name, values in component_arrays.items():
            if len(values) == len(H_values) and len(values) > 1:
                # Check for zero variance to avoid NaN
                h_std = np.std(H_values)
                v_std = np.std(values)
                if h_std > 0 and v_std > 0:
                    corr = np.corrcoef(H_values, values)[0, 1]
                else:
                    corr = 0.0
                correlations[name] = corr

        # Detect patterns
        patterns = {}
        patterns["oscillatory_dominant"] = np.std(component_arrays["oscillatory"]) > np.std(H_values) * 0.5
        patterns["stochastic_dominant"] = np.std(component_arrays["stochastic"]) > np.std(H_values) * 0.3
        
        # Check for zero variance before computing correlation
        if len(t_values) > 1 and len(component_arrays["integral"]) == len(t_values):
            t_std = np.std(t_values)
            i_std = np.std(component_arrays["integral"])
            if t_std > 0 and i_std > 0:
                patterns["integral_growing"] = np.corrcoef(t_values, component_arrays["integral"])[0, 1] > 0.5
            else:
                patterns["integral_growing"] = False
        else:
            patterns["integral_growing"] = False
            
        patterns["feedback_stable"] = np.std(component_arrays["delayed_feedback"]) < np.std(H_values) * 0.2

        return stats, correlations, patterns

    def plot_results(self, t_values: np.ndarray, H_values: np.ndarray, components: Dict):
        """Create comprehensive visualization of the simulation results."""
        print("\n📈 Generating Visualizations...")

        # Set up the plotting style with a version-tolerant choice
        try:
            plt.style.use("seaborn-v0_8-darkgrid")
        except (OSError, ValueError):
            # Fallback for environments where the version-specific style is unavailable
            try:
                plt.style.use("seaborn-darkgrid")
            except (OSError, ValueError):
                # Final fallback to default style
                pass
        plt.figure(figsize=(20, 16))

        # Main Hamiltonian evolution
        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(t_values, H_values, "b-", linewidth=2, label="H(t)")
        ax1.set_title("Complex Hamiltonian Evolution", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Time t")
        ax1.set_ylabel("H(t)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Component breakdown
        ax2 = plt.subplot(3, 2, 2)
        component_arrays = {k: np.array(v) for k, v in components.items()}
        for name, values in component_arrays.items():
            if len(values) == len(t_values):
                ax2.plot(t_values, values, label=name.replace("_", " ").title(), alpha=0.7)
        ax2.set_title("Component Contributions", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Time t")
        ax2.set_ylabel("Component Value")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Phase space (H vs dH/dt)
        ax3 = plt.subplot(3, 2, 3)
        dH_dt = np.gradient(H_values, t_values)
        ax3.scatter(H_values, dH_dt, c=t_values, cmap="viridis", alpha=0.6, s=10)
        ax3.set_title("Phase Space: H vs dH/dt", fontsize=14, fontweight="bold")
        ax3.set_xlabel("H(t)")
        ax3.set_ylabel("dH/dt")
        ax3.grid(True, alpha=0.3)

        # Power spectrum
        ax4 = plt.subplot(3, 2, 4)
        fft_H = np.fft.fft(H_values)
        freqs = np.fft.fftfreq(len(H_values), t_values[1] - t_values[0])
        positive_freqs = freqs > 0
        ax4.semilogy(freqs[positive_freqs], np.abs(fft_H[positive_freqs]), "r-")
        ax4.set_title("Power Spectrum", fontsize=14, fontweight="bold")
        ax4.set_xlabel("Frequency")
        ax4.set_ylabel("Power (log scale)")
        ax4.grid(True, alpha=0.3)

        # Component correlations heatmap
        ax5 = plt.subplot(3, 2, 5)
        component_names = list(component_arrays.keys())
        corr_matrix = np.zeros((len(component_names), len(component_names)))

        for i, name1 in enumerate(component_names):
            for j, name2 in enumerate(component_names):
                x = component_arrays[name1]
                y = component_arrays[name2]
                if len(x) == len(y) and len(x) > 1:
                    # Avoid np.corrcoef returning NaN when one of the arrays has zero variance
                    x_std = np.std(x)
                    y_std = np.std(y)
                    if x_std > 0 and y_std > 0:
                        corr = np.corrcoef(x, y)[0, 1]
                    else:
                        corr = 0.0
                    corr_matrix[i, j] = corr

        im = ax5.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
        ax5.set_xticks(range(len(component_names)))
        ax5.set_yticks(range(len(component_names)))
        ax5.set_xticklabels(
            [name.replace("_", " ").title() for name in component_names], rotation=45
        )
        ax5.set_yticklabels([name.replace("_", " ").title() for name in component_names])
        ax5.set_title("Component Correlations", fontsize=14, fontweight="bold")
        plt.colorbar(im, ax=ax5)

        # Statistical summary
        ax6 = plt.subplot(3, 2, 6)
        stats_text = f"""
        Statistical Summary:
        
        Mean: {np.mean(H_values):.3f}
        Std Dev: {np.std(H_values):.3f}
        Min: {np.min(H_values):.3f}
        Max: {np.max(H_values):.3f}
        Range: {np.max(H_values) - np.min(H_values):.3f}
        
        Total Oscillations: {len([i for i in range(1, len(H_values)) if (H_values[i] - H_values[i-1]) * (H_values[i-1] - H_values[i-2]) < 0])}
        """
        ax6.text(
            0.1,
            0.9,
            stats_text,
            transform=ax6.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        )
        ax6.set_title("Statistical Summary", fontsize=14, fontweight="bold")
        ax6.axis("off")

        plt.tight_layout()
        plt.savefig("hamiltonian_simulation_results.png", dpi=300, bbox_inches="tight")
        plt.show()

        print("📊 Visualizations saved as 'hamiltonian_simulation_results.png'")


def run_comprehensive_simulation():
    """Run a comprehensive simulation with analysis."""
    print("🎯 Complex Hamiltonian Interaction Analysis")
    print("=" * 50)

    # Create simulator with default parameters
    simulator = ComplexHamiltonianSimulator()

    # Run simulation
    t_values, H_values, components = simulator.simulate()

    # Analyze behavior
    stats, correlations, patterns = simulator.analyze_behavior(t_values, H_values, components)

    # Print analysis results
    print("\n📊 Analysis Results:")
    print("-" * 30)
    print(f"Mean H(t): {stats['mean']:.3f}")
    print(f"Standard Deviation: {stats['std']:.3f}")
    print(f"Range: {stats['range']:.3f}")
    print(f"Variance: {stats['variance']:.3f}")

    print("\n🔗 Component Correlations with H(t):")
    for component, corr in correlations.items():
        print(f"  {component.replace('_', ' ').title()}: {corr:.3f}")

    print("\n🎭 Detected Patterns:")
    for pattern, detected in patterns.items():
        status = "✅" if detected else "❌"
        print(f"  {status} {pattern.replace('_', ' ').title()}")

    # Create visualizations
    simulator.plot_results(t_values, H_values, components)

    return simulator, t_values, H_values, components


if __name__ == "__main__":
    # Run the comprehensive simulation
    simulator, t_values, H_values, components = run_comprehensive_simulation()

    print("\n🎉 Simulation Complete!")
    print("The complex Hamiltonian shows emergent behavior from the interaction of:")
    print("• Oscillatory and decaying components")
    print("• Nonlinear integral memory effects")
    print("• Polynomial, periodic, and logarithmic drifts")
    print("• Delayed nonlinear feedback")
    print("• State-dependent stochasticity")
    print("• External input coupling")
