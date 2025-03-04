# Gate Manager

A Python package for controlling and managing gate voltages in quantum devices.

## Features

- **Easy Voltage Control**
  Set, ramp, and read gate voltages through simple Python calls.
- **1D & 2D Sweeps**
  Automate complex voltage sweeps across multiple gates, logging data for analysis.
- **Instrument Agnostic**
  Easily extend or adapt the package to support different hardware backends (e.g., Nanonis, Zurich).
- **Integration & Logging**
  Built-in logging of experimental parameters and sweep data for reproducible research.

## Installation

You can install `gate-manager` using `pip`:

```bash
pip install git+https://github.com/chenx820/gate-manager.git
```

or

```bash
pip install gate-manager
```

## Development Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/chenx820/gate-manager.git
cd gate-manager
pip install -e .
```

This way, changes to the source code will immediately be reflected in your environment without reinstallation.

## Usage

Below is a quick example showing how you might use the package to perform a simple 1D sweep. This is a high-level illustration; adjust names and parameters to match your hardware setup.

```bash
from gate_manager.gate import Gate, GatesGroup
from gate_manager.sweeper import Sweeper

# Suppose you have Gate instances connected to your hardware:
output_gate_1 = Gate(...)
output_gate_2 = Gate(...)
input_gate_1  = Gate(...)

# Group them as needed:
outputs = GatesGroup([output_gate_1, output_gate_2])
inputs  = GatesGroup([input_gate_1])

# Create a Sweeper instance
sweeper = Sweeper(outputs=outputs, inputs=inputs, temperature="300K", device="QuantumDevice1")

# Prepare an initial state for the outputs
initial_state = [
    (output_gate_1, 0.0),
    (output_gate_2, 0.0),
]

# Perform a 1D sweep from 0 V to 1 V in steps of 0.1 V
sweeper.sweep1D(
    swept_outputs=outputs,
    measured_inputs=inputs,
    start_voltage=0.0,
    end_voltage=1.0,
    step=0.1,
    initial_state=initial_state,
    comments="MyFirstSweep"
)
```

Check out the `example/` folder to see more advanced usage.

## Testing

We use pytest for unit tests. After cloning the repository and installing dependencies:

```bash
cd gate-manager
pytest
```

This will run all tests in the `tests/` folder (e.g., `test_gate.py`, `test_sweep.py`, etc.). If you want to see a coverage report, install pytest-cov:

```bash
pip install pytest-cov
pytest --cov=gate_manager --cov-report=term-missing
```

## Contributing

Contributions are welcome! Whether you find a bug, have a feature request, or want to add new hardware support, here are a few ways to get involved:

1. **Fork & Clone:** Fork the repository, then clone it locally.
2. **Create a Branch:** Make a feature or fix branch, e.g. git checkout -b feature/my-new-feature.
3. **Make Changes:** Update the code, add tests, and ensure all tests pass.
4. **Open a Pull Request:** Submit your PR, describing your changes and why they’re needed.

Please folow PEP 8 style guidelines and ensure that all existing tests pass before submitting a pull request.

## License

This project is licensed under the MIT License.
