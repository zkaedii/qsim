# Mathematical Formulation

## Multi-Component Stochastic Oscillator (MCSO)

### System Definition

The Multi-Component Stochastic Oscillator is defined by the state equation:

$$X(t) = S(t) + I(t) + D(t) + M(t) + N(t) + U(t)$$

where each component contributes different dynamical characteristics.

---

## Component Definitions

### 1. Oscillatory Term S(t)

The oscillatory component consists of multiple damped sinusoidal modes:

$$S(t) = \sum_{i=0}^{n-1} \left[ A_i(t) \sin(B_i(t) \cdot t + \phi_i) + C_i e^{-D_i t} \right]$$

**Parameters:**
- $A_i(t) = A_{base} + A_{mod} \sin(\omega_{mod} t)$ — Time-varying amplitude
- $B_i = B_{base} + B_{scale} \cdot i$ — Frequency (harmonic structure)
- $\phi_i = \pi / (i + 1)$ — Phase offset
- $C_i$ — Decay coefficient
- $D_i = D_{base} + D_{scale} \cdot i$ — Decay rate

**Interpretation:** Models periodic phenomena with multiple frequency components and transient decay. Common in:
- Mechanical oscillations
- Biological rhythms
- Market cycles

---

### 2. Integral Term I(t)

A nonlinear integral involving activation function:

$$I(t) = \int_0^t \sigma(a(x - x_0)^2 + b) \cdot f(x) \cdot g'(x) \, dx$$

where $\sigma(\cdot)$ is the softplus activation:

$$\sigma(z) = \log(1 + e^z)$$

**Default functions:**
- $f(x) = \cos(x)$
- $g'(x) = -\sin(x)$

**Interpretation:** Captures cumulative effects with nonlinear weighting. The softplus provides:
- Smooth, differentiable nonlinearity
- Positive-only contributions
- Natural saturation behavior

---

### 3. Drift Term D(t)

Deterministic trend component:

$$D(t) = \alpha_0 t^2 + \alpha_1 \sin(2\pi t) + \alpha_2 \log(1 + t)$$

**Parameters:**
- $\alpha_0$ — Quadratic drift (acceleration)
- $\alpha_1$ — Periodic component
- $\alpha_2$ — Logarithmic growth

**Interpretation:** Models:
- Long-term trends ($\alpha_0$)
- Seasonal patterns ($\alpha_1$)
- Diminishing returns ($\alpha_2$)

---

### 4. Memory Term M(t)

Sigmoid-gated feedback with discrete delay:

$$M(t) = \eta \cdot X(t - \tau) \cdot \sigma_s(\gamma \cdot X(t - \tau))$$

where $\sigma_s(\cdot)$ is the sigmoid function:

$$\sigma_s(z) = \frac{1}{1 + e^{-z}}$$

**Parameters:**
- $\eta$ — Memory strength
- $\tau$ — Time delay
- $\gamma$ — Sigmoid sensitivity

**Properties:**
- Bounded feedback: $|M(t)| \leq \eta \cdot |X(t-\tau)|$
- Asymmetric response for positive/negative states
- Self-regulating: large states are partially suppressed

**Interpretation:** Models:
- Momentum effects (positive $\gamma$)
- Mean reversion (negative $\gamma$)
- Hysteresis phenomena

---

### 5. Noise Term N(t)

State-dependent stochastic forcing:

$$N(t) = \sigma \cdot \varepsilon(t) \cdot \sqrt{1 + \beta |X(t-1)|}$$

where $\varepsilon(t) \sim \mathcal{N}(0, 1)$ i.i.d.

**Parameters:**
- $\sigma$ — Base noise scale
- $\beta$ — State-noise coupling

**Properties:**
- When $\beta = 0$: standard additive noise
- When $\beta > 0$: volatility clustering (larger states → more noise)
- Variance: $\text{Var}[N(t)] = \sigma^2 (1 + \beta |X(t-1)|)$

**Interpretation:** Models:
- Measurement uncertainty
- Environmental fluctuations
- GARCH-like volatility effects

---

### 6. Control Term U(t)

External control input:

$$U(t) = \delta \cdot u(t)$$

**Parameters:**
- $\delta$ — Control gain
- $u(t)$ — Control function (user-defined or default sinusoid)

**Default:** $u(t) = \sin(\omega_c t)$

**Interpretation:** Allows external forcing for:
- System identification
- Control experiments
- Scenario analysis

---

## Activation Functions

### Softplus

$$\sigma(z) = \log(1 + e^z) \approx \begin{cases} z & z \gg 0 \\ e^z & z \ll 0 \end{cases}$$

**Properties:**
- Smooth approximation to ReLU
- $\sigma(z) > 0$ for all $z$
- $\sigma'(z) = \sigma_s(z)$ (sigmoid)

### Sigmoid

$$\sigma_s(z) = \frac{1}{1 + e^{-z}}$$

**Properties:**
- Output in $(0, 1)$
- $\sigma_s(0) = 0.5$
- $\sigma_s(-z) = 1 - \sigma_s(z)$

---

## Numerical Considerations

### Stability

1. **Time clipping:** $t \leftarrow \min(t, t_{max})$ prevents overflow in exponential terms
2. **Activation clipping:** Input to softplus/sigmoid clipped to $[-500, 500]$
3. **Output clipping:** Final state bounded by `clip_bounds`

### Integration

- Integral term uses adaptive quadrature (scipy.integrate.quad)
- Integration range limited to `integral_limit` for efficiency
- Subdivision limit controls accuracy vs. speed tradeoff

### Reproducibility

- Random number generator: `numpy.random.Generator`
- Seed propagation: consistent seeding for reproducible experiments

---

## Parameter Summary

| Parameter | Symbol | Default | Range | Effect |
|-----------|--------|---------|-------|--------|
| n_components | $n$ | 5 | [1, ∞) | Spectral complexity |
| amplitude_base | $A_{base}$ | 1.0 | ℝ | Oscillation magnitude |
| amplitude_modulation | $A_{mod}$ | 0.1 | [0, ∞) | AM depth |
| frequency_base | $B_{base}$ | 1.0 | ℝ₊ | Fundamental frequency |
| frequency_scaling | $B_{scale}$ | 0.1 | ℝ | Harmonic spacing |
| decay_base | $D_{base}$ | 0.05 | ℝ₊ | Transient decay |
| noise_scale | $\sigma$ | 0.2 | [0, ∞) | Noise intensity |
| noise_state_coupling | $\beta$ | 0.3 | [0, ∞) | Volatility clustering |
| memory_strength | $\eta$ | 1.0 | ℝ | Feedback strength |
| memory_delay | $\tau$ | 1.0 | ℝ₊ | Delay time |
| memory_sensitivity | $\gamma$ | 2.0 | ℝ | Gate sharpness |
| control_gain | $\delta$ | 0.1 | ℝ | Control influence |

---

## Theoretical Properties

### Stationarity

For sufficiently small $\eta$ and $\alpha_0 = 0$, the process is asymptotically stationary.

**Condition:** $|\eta| < 1$ and drift bounded.

### Ergodicity

Under stationarity conditions, time averages converge to ensemble averages:

$$\lim_{T \to \infty} \frac{1}{T} \int_0^T f(X(t)) \, dt = \mathbb{E}[f(X)]$$

### Mixing

The memory term introduces temporal dependence. Mixing rate depends on:
- Memory delay $\tau$
- Memory strength $\eta$
- Sigmoid sensitivity $\gamma$

---

## Extensions

### Multi-Scale Memory

Replace single-delay memory with weighted sum:

$$M(t) = \sum_j \eta_j \cdot e^{-(t-t_j)/\tau_j} \cdot X(t_j)$$

### Jump-Diffusion Noise

Replace Gaussian noise with compound Poisson:

$$N(t) = \sigma_c \varepsilon_c(t) + \sum_{k=1}^{N_P(t)} J_k$$

where $N_P(t)$ is Poisson with rate $\lambda$ and $J_k \sim \mathcal{N}(\mu_J, \sigma_J^2)$.

### Adaptive Parameters

Allow parameters to evolve based on state:

$$\eta(t+1) = \eta(t) + \alpha (X(t) - \hat{X}(t)) X(t-\tau)$$

---

## References

1. Gardiner, C. W. (2009). *Stochastic Methods*. Springer.
2. Risken, H. (1996). *The Fokker-Planck Equation*. Springer.
3. Kloeden, P. E., & Platen, E. (1992). *Numerical Solution of SDEs*. Springer.
