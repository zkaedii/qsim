# Meta-Engine: Multiform Signal Processing Chassis

## Overview

The Meta-Signal Processing Chassis is a self-adaptive engine for multiform signal processing that embodies polymorphic, metamorphic, and adversarial-resilient characteristics. It provides a flexible framework for composing complex signal processing pipelines from modular subsystems.

## Architecture

### Core Principles

The meta-engine embodies multiple morphic properties:

- **Polymorphic**: Adapts across diverse object structures through abstract base classes
- **Metamorphic**: Evolves internal shape and function during runtime via adaptation
- **Oglimorphic**: Bounded ambiguity with robust discrete exception handling
- **Chameleon**: Context-aware self-adjustment without brittle failures
- **Adaptive**: Dynamic pathway modification under feedback
- **Adversarial-Resilient**: Deception-aware processing with defensive strategies
- **Traversal-Oriented**: Fluid navigation of hierarchical structures
- **Nanomorphic**: Tiny modularized units enabling scalable complexity
- **Homomorphic**: End-state transformation preserving structure
- **Holomorphic**: Entire continuity with harmonic completeness

### Chassis Layers

The processing flow follows a cascade architecture:

```
Oscillation → Spectrum → Modulation → Synchronization → Recovery
```

Each computational subsystem morphs polymorphically while maintaining core trust and harmonic coherence.

## Components

### MetaSignalChassis

The central orchestrator for signal processing pipelines.

**Key Features:**
- Polymorphic subsystem registration with priorities
- Adversarial noise reduction via softplus activation
- Automatic error recovery with fallback mechanisms
- Execution history tracking
- Metamorphic adaptation support

**Basic Usage:**

```python
from hmodelz.engines import MetaSignalChassis, create_oscillator_subsystem

# Create chassis
chassis = MetaSignalChassis(
    adversarial_threshold=1.0,
    enable_recovery=True
)

# Register subsystems
oscillator = create_oscillator_subsystem([1.0, 2.0], [1.0, 0.5])
chassis.register_subsystem("oscillator", oscillator, priority=1)

# Execute pipeline
result = chassis.dispatch({'t': np.linspace(0, 10, 100)})
```

### Subsystem Base Class

Abstract base class for all subsystem implementations.

**Required Methods:**
- `execute(context)`: Main processing logic
- `adapt(feedback)`: Optional metamorphic adaptation

**Example Implementation:**

```python
from hmodelz.engines import Subsystem, SubsystemConfig

class CustomSubsystem(Subsystem):
    def execute(self, context):
        # Your processing logic
        data = context.get('input_data')
        processed = self.process(data)
        return {'output': processed}
    
    def process(self, data):
        # Implementation details
        return data * 2
```

### MultiOscillator

Built-in subsystem for multi-frequency oscillator synthesis.

**Features:**
- Superposition of multiple oscillators
- Harmonic synthesis with phase control
- Exponential decay modeling
- Configurable amplitude and frequency

**Usage:**

```python
from hmodelz.engines import create_oscillator_subsystem

# Create multi-oscillator
oscillator = create_oscillator_subsystem(
    frequencies=[1.0, 2.0, 3.0],      # Frequencies (Hz)
    amplitudes=[1.0, 0.5, 0.25],      # Amplitudes
    phase_offsets=[0, π/4, π/2],      # Phase shifts (rad)
    decay_constants=[0, 0.1, 0.2]     # Decay rates
)

# Execute
result = oscillator.execute({'t': time_array})
signal = result['signal']
```

## Adversarial Noise Handling

The chassis includes built-in adversarial noise reduction using softplus activation:

```python
# Clean noisy signal
noisy_signal = np.array([1.0, 100.0, 2.0, -50.0, 3.0])
cleaned = chassis.handle_adversarial_context(
    noisy_signal,
    noise_model=0.0
)
```

The softplus function stabilizes adversarial inputs:

```
softplus(x) = log(1 + exp(x - model))
```

This reduces extreme values while preserving legitimate signals.

## Metamorphic Adaptation

Subsystems can evolve during runtime through adaptation:

```python
# Create metamorphic subsystem
config = SubsystemConfig(
    name="adaptive",
    metamorphic=True
)
subsystem = MySubsystem(config)

# Adapt based on feedback
feedback = {
    'learning_rate': 0.01,
    'optimization': 'enabled'
}
subsystem.adapt(feedback)

# State is updated
print(subsystem.state)  # {'learning_rate': 0.01, 'optimization': 'enabled'}
```

## Priority-Based Execution

Subsystems execute in priority order (lower numbers first):

```python
chassis.register_subsystem("preprocessing", preproc, priority=1)
chassis.register_subsystem("main_process", main, priority=2)
chassis.register_subsystem("postprocessing", postproc, priority=3)

# Executes: preprocessing → main_process → postprocessing
chassis.dispatch(initial_context)
```

## Error Recovery

The chassis supports automatic error recovery:

```python
# With recovery (default)
chassis = MetaSignalChassis(enable_recovery=True)
result = chassis.dispatch(context)  # Continues on errors

# Without recovery
chassis = MetaSignalChassis(enable_recovery=False)
result = chassis.dispatch(context)  # Raises on first error
```

Failed subsystems return error information:

```python
{
    'subsystem_name': {
        'error': 'Error message',
        'recovered': True
    }
}
```

## Examples

### Example 1: Basic Oscillator

```python
from hmodelz.engines import create_default_chassis, create_oscillator_subsystem
import numpy as np

# Setup
chassis = create_default_chassis()
oscillator = create_oscillator_subsystem([1.0], [1.0])
chassis.register_subsystem("osc", oscillator)

# Execute
t = np.linspace(0, 10, 1000)
result = chassis.dispatch({'t': t})
signal = result['osc']['signal']
```

### Example 2: Harmonic Synthesis

```python
# Create harmonic series
fundamental = 440.0  # A4 note
harmonics = [fundamental * i for i in [1, 2, 3, 4, 5]]
amplitudes = [1.0, 0.5, 0.33, 0.25, 0.2]

# Build oscillator
oscillator = create_oscillator_subsystem(harmonics, amplitudes)

# Generate tone
t = np.linspace(0, 1, 44100)  # 1 second at 44.1kHz
result = oscillator.execute({'t': t})
audio_signal = result['signal']
```

### Example 3: Pipeline Composition

```python
# Create processing pipeline
chassis = MetaSignalChassis()

# Stage 1: Generate base signal
base_osc = create_oscillator_subsystem([1.0], [1.0])
chassis.register_subsystem("base", base_osc, priority=1)

# Stage 2: Add harmonics
harmonics_osc = create_oscillator_subsystem([2.0, 3.0], [0.5, 0.25])
chassis.register_subsystem("harmonics", harmonics_osc, priority=2)

# Stage 3: Apply modulation
modulator = create_oscillator_subsystem([0.1], [0.2])
chassis.register_subsystem("modulation", modulator, priority=3)

# Execute full pipeline
result = chassis.dispatch({'t': time_array})
```

## API Reference

### MetaSignalChassis

#### Constructor

```python
MetaSignalChassis(
    adversarial_threshold: float = 1.0,
    enable_recovery: bool = True
)
```

#### Methods

- `register_subsystem(name, subsystem, priority=0)`: Register a subsystem
- `unregister_subsystem(name)`: Remove a subsystem
- `dispatch(initial_context=None)`: Execute all subsystems
- `handle_adversarial_context(inputs, noise_model=0.0)`: Clean noisy signals
- `reset()`: Reset chassis to initial state
- `get_execution_summary()`: Get execution statistics

### MultiOscillator

#### Constructor

```python
MultiOscillator(config: Optional[SubsystemConfig] = None)
```

#### Methods

- `add_oscillator(frequency, amplitude, phase_offset=0.0, decay_constant=0.0)`: Add oscillator model
- `compute_superposition(t)`: Compute combined signal
- `execute(context)`: Process in chassis context

### Factory Functions

```python
create_default_chassis() -> MetaSignalChassis
create_oscillator_subsystem(
    frequencies: List[float],
    amplitudes: List[float],
    phase_offsets: Optional[List[float]] = None,
    decay_constants: Optional[List[float]] = None
) -> MultiOscillator
```

## Performance Considerations

- **Memory Efficiency**: Use circular buffers for long-running simulations
- **Vectorization**: NumPy operations are vectorized for performance
- **Priority Ordering**: O(n log n) sorting of subsystems at dispatch
- **Error Handling**: Try-except overhead only when recovery is enabled

## Testing

Comprehensive test suite with 39 tests covering:

- Subsystem configuration and creation
- Polymorphic registration and execution
- Adversarial noise handling
- Metamorphic adaptation
- Priority-based dispatch
- Error recovery mechanisms
- Integration scenarios

Run tests:

```bash
pytest tests/unit/test_meta_signal_chassis.py -v
```

## Future Extensions

Potential areas for expansion:

1. **Spectrum Analysis**: FFT-based frequency domain processing
2. **Modulation**: AM/FM/PM modulation subsystems
3. **Synchronization**: Phase-locked loop subsystems
4. **Filtering**: Adaptive filter bank subsystems
5. **Machine Learning**: Neural network integration for signal prediction

## References

- Softplus activation: `log(1 + exp(x))`
- Oscillator superposition: `Σ Aᵢ sin(ωᵢt + φᵢ) exp(-Dᵢt)`
- Polymorphic design patterns in signal processing
- Adversarial robustness in computational systems

## License

MIT License - See repository LICENSE file for details.
