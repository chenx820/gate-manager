import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for tests
import matplotlib.pyplot as plt
import pytest

from gate_manager.sweeper import Sweeper

# ---------------------------
# Dummy Classes for Testing
# ---------------------------

class DummyLine:
    """A simple dummy line with a label."""
    def __init__(self, label):
        self.label = label

class DummyGate:
    """
    A dummy gate that simulates a Gate object.
    It supports setting/reading a voltage, checking target voltage, and reading a current.
    """
    def __init__(self, label, init_voltage=0.0):
        self.lines = [DummyLine(label)]
        self.current_voltage = init_voltage
        # Create a dummy source with required attributes.
        DummySource = type("DummySource", (), {})  
        self.source = DummySource()
        self.source.label = label + " Source"
        self.source.write_index = 1
        self.source.read_index = 1

    def voltage(self, target_voltage=None, is_wait=True):
        """
        If target_voltage is provided, set the gate voltage.
        Otherwise, return the current voltage.
        """
        if target_voltage is None:
            return self.current_voltage
        else:
            self.current_voltage = target_voltage
            return self.current_voltage

    def is_at_target_voltage(self, target_voltage, tolerance=1e-6):
        """Return True if the current voltage is within tolerance of the target."""
        return abs(self.current_voltage - target_voltage) < tolerance

    def read_current(self, amplification):
        """
        For testing, return a current reading that is twice the current voltage.
        (This value will be scaled later by the Sweeper.)
        """
        return self.current_voltage * 2

    def turn_off(self, is_wait=True):
        """Simulate turning off the gate by setting its voltage to 0."""
        self.current_voltage = 0.0

class DummyGatesGroup:
    """
    A dummy group of gates that mimics the behavior of a GatesGroup.
    It supports setting voltage on all contained gates and turning them off.
    """
    def __init__(self, gates):
        self.gates = gates

    def voltage(self, target_voltage, is_wait=True):
        for gate in self.gates:
            gate.voltage(target_voltage, is_wait)

    def turn_off(self, is_wait=True):
        for gate in self.gates:
            gate.turn_off()

# A dummy Visualizer for sweep2D testing.
class DummyVisualizer:
    def viz2D(self, filename):
        DummyVisualizer.called = True

DummyVisualizer.called = False

# ---------------------------
# Fixtures
# ---------------------------

@pytest.fixture
def dummy_gate():
    """Return a dummy gate with an initial voltage of 0."""
    return DummyGate("TestGate", init_voltage=0.0)

@pytest.fixture
def dummy_gate_group():
    """Return a dummy gates group containing one dummy gate."""
    gate = DummyGate("GroupGate", init_voltage=0.0)
    return DummyGatesGroup([gate])

@pytest.fixture
def dummy_outputs():
    """
    Return a dummy outputs group.
    (This group is used by the Sweeper instance for logging initial output voltages.)
    """
    # Create one dummy gate with an arbitrary starting voltage.
    gate = DummyGate("OutputGate", init_voltage=1.0)
    return DummyGatesGroup([gate])

@pytest.fixture
def dummy_measured_inputs():
    """
    Return a dummy measured inputs group.
    The first gate in this group will be used to read current.
    """
    gate = DummyGate("MeasuredGate", init_voltage=0.0)
    return DummyGatesGroup([gate])

# ---------------------------
# Tests for Internal Methods
# ---------------------------

def test_set_units():
    """
    Test that _set_units correctly sets voltage and current scales.
    """
    sweeper = Sweeper()
    sweeper._set_units(voltage_unit='V', current_unit='uA')
    # For 'V' and 'uA', scales should be 1.
    assert sweeper.voltage_scale == 1
    assert sweeper.current_scale == 1

def test_set_gates_group_label():
    """
    Test that _set_gates_group_label returns a combined label from all gate lines.
    """
    # Create two dummy gates with distinct labels.
    gate1 = DummyGate("GateA")
    gate2 = DummyGate("GateB")
    group = DummyGatesGroup([gate1, gate2])
    sweeper = Sweeper()
    label = sweeper._set_gates_group_label(group)
    # Expect the label to be "GateA & GateB"
    assert label == "GateA & GateB"

# ---------------------------
# Tests for Sweep Methods
# ---------------------------

def test_sweep1D(tmp_path, monkeypatch):
    """
    Test the 1D sweep method.
    
    This test sets up dummy groups for swept outputs, measured inputs, and outputs.
    It runs a minimal sweep (from 0V to 0.1V in one step) and checks that the recorded
    voltages and currents are as expected.
    """
    # Redirect file I/O to the temporary directory.
    monkeypatch.setattr(os, "getcwd", lambda: str(tmp_path))
    
    # Create dummy groups.
    # Swept outputs: one dummy gate.
    swept_gate = DummyGate("SweptGate", init_voltage=0.0)
    swept_outputs = DummyGatesGroup([swept_gate])
    
    # Measured inputs: one dummy gate.
    measured_gate = DummyGate("MeasuredGate", init_voltage=0.0)
    measured_inputs = DummyGatesGroup([measured_gate])
    
    # Outputs (used for logging initial voltages): one dummy gate.
    output_gate = DummyGate("OutputGate", init_voltage=1.0)
    outputs = DummyGatesGroup([output_gate])
    
    # Define initial state: set the output gate voltage to 0.
    initial_state = [(output_gate, 0.0)]
    
    # Create a Sweeper instance with dummy outputs and measured inputs.
    sweeper = Sweeper(outputs=outputs, inputs=measured_inputs, amplification=1, temperature="CT", device="DummyDevice")
    
    # Use a small sweep: from 0.0V to 0.1V with one step.
    start_voltage = 0.0
    end_voltage = 0.1
    step = 0.1
    
    # Run the sweep1D method.
    sweeper.sweep1D(
        swept_outputs=swept_outputs,
        measured_inputs=measured_inputs,
        start_voltage=start_voltage,
        end_voltage=end_voltage,
        step=step,
        initial_state=initial_state,
        voltage_unit='V',
        current_unit='uA',
        comments="TestSweep1D",
        ax2=None,
        is_2d_sweep=False
    )
    # Expect two iterations: one at 0.0V and one at 0.1V.
    expected_voltages = [0.0, 0.1]
    # Since voltage_scale is 1 for 'V'
    assert np.allclose(sweeper.voltages, expected_voltages, atol=1e-6)
    # Check that currents were recorded (length should match voltages)
    assert len(sweeper.currents) == len(sweeper.voltages)

def test_sweepTime(tmp_path, monkeypatch):
    """
    Test the time-based sweep method.
    
    This test sets up a dummy measured inputs group and runs a short time sweep.
    It verifies that current measurements are recorded.
    """
    # Redirect file I/O to the temporary directory.
    monkeypatch.setattr(os, "getcwd", lambda: str(tmp_path))
    
    # Create dummy measured inputs: one dummy gate.
    measured_gate = DummyGate("MeasuredGate", init_voltage=0.0)
    measured_inputs = DummyGatesGroup([measured_gate])
    
    # Create a Sweeper instance (outputs not used in sweepTime here).
    sweeper = Sweeper(outputs=DummyGatesGroup([]), inputs=measured_inputs, amplification=1, temperature="CT", device="DummyDevice")
    
    total_time = 0.5   # seconds
    time_step = 0.2    # seconds
    
    # Run the time sweep.
    sweeper.sweepTime(
        measured_inputs=measured_inputs,
        total_time=total_time,
        time_step=time_step,
        initial_state=[(measured_gate, 0.0)],
        comments="TestSweepTime"
    )
    # Check that at least one current measurement was recorded.
    assert len(sweeper.currents) >= 1
    # Also, the last recorded current should be a float.
    assert isinstance(sweeper.currents[-1], float)

def test_sweep2D(tmp_path, monkeypatch):
    """
    Test the 2D sweep method.
    
    This test sets up dummy groups for X and Y swept outputs and measured inputs.
    It also monkeypatches the Visualizer to avoid generating an actual plot.
    The test verifies that the method runs without error and calls the viz2D method.
    """
    # Redirect file I/O to the temporary directory.
    monkeypatch.setattr(os, "getcwd", lambda: str(tmp_path))
    
    # Monkeypatch the Visualizer in the sweeper module to use our DummyVisualizer.
    monkeypatch.setattr("sweeper.Visualizer", lambda: DummyVisualizer())
    
    # Create dummy groups for X and Y swept outputs and measured inputs.
    x_gate = DummyGate("XGate", init_voltage=0.0)
    X_swept_outputs = DummyGatesGroup([x_gate])
    
    y_gate = DummyGate("YGate", init_voltage=0.0)
    Y_swept_outputs = DummyGatesGroup([y_gate])
    
    measured_gate = DummyGate("MeasuredGate", init_voltage=0.0)
    measured_inputs = DummyGatesGroup([measured_gate])
    
    # Define an initial state (for X sweep) with one tuple.
    initial_state = [(x_gate, 0.0)]
    
    # Create a Sweeper instance.
    sweeper = Sweeper(outputs=DummyGatesGroup([]), inputs=measured_inputs, amplification=1, temperature="25C", device="DummyDevice")
    
    # Use a very small 2D sweep.
    X_start_voltage = 0.0
    X_end_voltage = 0.1
    X_step = 0.1
    Y_start_voltage = 0.0
    Y_end_voltage = 0.1
    Y_step = 0.1
    
    # Run the 2D sweep.
    sweeper.sweep2D(
        X_swept_outputs=X_swept_outputs,
        X_start_voltage=X_start_voltage,
        X_end_voltage=X_end_voltage,
        X_step=X_step,
        Y_swept_outputs=Y_swept_outputs,
        Y_start_voltage=Y_start_voltage,
        Y_end_voltage=Y_end_voltage,
        Y_step=Y_step,
        measured_inputs=measured_inputs,
        initial_state=initial_state,
        voltage_unit='V',
        current_unit='uA',
        comments="TestSweep2D"
    )
    # Verify that the DummyVisualizer's viz2D method was called.
    assert DummyVisualizer.called is True
