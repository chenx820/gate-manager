# tests/test_sweeper.py

import os
import time
from decimal import Decimal

import numpy as np
import pytest

# Import tqdm from the package (for monkeypatching, we'll override the attribute in the module)
from tqdm import tqdm

# Correct import to match your module name
from gate_manager.sweeper import Sweeper

# ---------------------------------------------------------------------
# Dummy classes to simulate Gate and GatesGroup behavior for testing.
# These dummy classes mimic the minimal interface used by Sweeper.
# ---------------------------------------------------------------------

class DummyGate:
    def __init__(self, label, initial_voltage=0.0):
        self._voltage = float(initial_voltage)
        self.label = label
        # Create a dummy source and a dummy line with the same label.
        self.source = type("DummySource", (), {"label": label})
        self.lines = [type("DummyLine", (), {"label": label})()]
    
    def voltage(self, target_voltage=None, is_wait=True):
        if target_voltage is not None:
            self._voltage = float(target_voltage)
        return self._voltage
    
    def is_at_target_voltage(self, voltage, tolerance=1e-6):
        return abs(self._voltage - float(voltage)) < tolerance
    
    def read_current(self, amplification):
        # Return a dummy current value.
        return 0.001
    
    def turn_off(self, is_wait=True):
        self._voltage = 0.0

class DummyGatesGroup:
    def __init__(self, gates):
        self.gates = gates
    
    def voltage(self, target_voltage, is_wait=True):
        for gate in self.gates:
            gate.voltage(target_voltage, is_wait)

# ---------------------------------------------------------------------
# Tests for label-setting methods.
# ---------------------------------------------------------------------

def test_set_gates_group_label():
    # Create two dummy gates with labels "A" and "B"
    gate1 = DummyGate("A")
    gate2 = DummyGate("B")
    group = DummyGatesGroup([gate1, gate2])
    sweeper = Sweeper()
    label = sweeper.set_gates_group_label(group)
    # Expect "A & B" since each gate returns its label.
    assert label == "A & B"

def test_set_gate_label():
    gate = DummyGate("X")
    sweeper = Sweeper()
    label = sweeper.set_gate_label(gate)
    assert label == "X"

# ---------------------------------------------------------------------
# Tests for filename generation and logging.
# ---------------------------------------------------------------------

def test_set_filename(tmp_path):
    # Create a Sweeper instance and set properties.
    sweeper = Sweeper()
    # Use the attribute for saving files (assume code now uses is_save_file consistently).
    sweeper.is_save_file = True
    sweeper.temperature = "CT"
    sweeper.x_label = "GateX"
    sweeper.y_label = "GateY"
    sweeper.comments = "test"
    # Override os.getcwd() to return the temporary directory.
    original_getcwd = os.getcwd
    os.getcwd = lambda: str(tmp_path)
    try:
        sweeper.set_filename()
        expected_filename = "300K_[GateY]_vs_[GateX]_test"
        # The filename may have a run counter appended if a file exists;
        # here we check that it starts with the expected string.
        assert sweeper.filename.startswith(expected_filename)
    finally:
        os.getcwd = original_getcwd

def test_log_params(tmp_path, monkeypatch):
    # Test that log_params writes expected information.
    sweeper = Sweeper()
    sweeper.temperature = "CT"
    sweeper.x_label = "GateX"
    sweeper.y_label = "GateY"
    sweeper.device = "Device1"
    sweeper.amplification = 1.0
    sweeper.slew_rate = 0.1
    sweeper.start_voltage = Decimal("0.0")
    sweeper.end_voltage = Decimal("1.0")
    sweeper.step = Decimal("0.1")
    # Create a dummy output gate.
    gate = DummyGate("A")
    gate._voltage = 0.5
    sweeper.outputs = DummyGatesGroup([gate])
    sweeper.filename = "testfile"
    # Redirect file writes to a temporary file.
    log_file = tmp_path / "log_test.txt"
    def fake_open(filename, mode):
        return log_file.open(mode)
    monkeypatch.setattr("builtins.open", fake_open)
    sweeper.log_params("voltage")
    content = log_file.read_text()
    # Instead of checking for "300K" (which is not printed), check for keys that should appear.
    assert "Device:" in content
    assert "Swept Gates:" in content

# ---------------------------------------------------------------------
# Tests for sweep1D.
# ---------------------------------------------------------------------

@pytest.fixture
def dummy_outputs_and_inputs():
    # Create one dummy output gate and one dummy input gate.
    gate_out = DummyGate("Out")
    gate_in = DummyGate("In")
    outputs = DummyGatesGroup([gate_out])
    inputs = DummyGatesGroup([gate_in])
    initial_state = [(gate_out, 0.0)]
    return outputs, inputs, initial_state

def dummy_tqdm(*args, **kwargs):
    # Return a dummy progress bar that does nothing.
    class DummyBar:
        def update(self, x):
            pass
        def close(self):
            pass
    return DummyBar()

def test_sweep1D(monkeypatch, dummy_outputs_and_inputs):
    outputs, inputs, initial_state = dummy_outputs_and_inputs
    sweeper = Sweeper(outputs=outputs, inputs=inputs, amplification=1.0, temperature="CT", device="Device1")
    sweeper.is_save_file = False
    # Override time.sleep to avoid delay.
    monkeypatch.setattr(time, "sleep", lambda x: None)
    # Override tqdm in the Sweeper module.
    monkeypatch.setattr("gate_manager.sweeper.tqdm", dummy_tqdm)
    # Run sweep1D with a small sweep range.
    sweeper.sweep1D(swept_outputs=outputs, measured_inputs=inputs,
                    start_voltage=0.0, end_voltage=0.2, step=0.1,
                    initial_state=initial_state, comments="1D_test", is_save_file=False)
    # After sweep, the arrays for voltages and currents should be nonempty.
    assert len(sweeper.voltages) > 0
    assert len(sweeper.currents) > 0

def test_sweep2D(monkeypatch, dummy_outputs_and_inputs):
    outputs, inputs, initial_state = dummy_outputs_and_inputs
    sweeper = Sweeper(outputs=outputs, inputs=inputs, amplification=1.0, temperature="CT", device="Device1")
    sweeper.is_save_file = False
    # Set Y_voltage so that the code in 2D mode can use it.
    sweeper.Y_voltage = 0.0
    monkeypatch.setattr(time, "sleep", lambda x: None)
    monkeypatch.setattr("gate_manager.sweeper.tqdm", dummy_tqdm)
    import matplotlib.pyplot as plt
    # Create a dummy axis for 2D sweep mode.
    dummy_fig, dummy_ax = plt.subplots(1, 1, figsize=(8, 6))
    # Call sweep1D with the dummy axis passed via the ax2 parameter.
    result = sweeper.sweep1D(
        swept_outputs=outputs,
        measured_inputs=inputs,
        start_voltage=0.0,
        end_voltage=0.2,
        step=0.1,
        initial_state=initial_state,
        comments="2D_test",
        is_save_file=False,
        is_2d_sweep=True,
        ax2=dummy_ax
    )
    # The function should return a tuple (voltages, currents)
    voltages, currents = result
    assert isinstance(voltages, list)
    assert isinstance(currents, list)

# ---------------------------------------------------------------------
# Test for sweepTime.
# ---------------------------------------------------------------------

def test_sweepTime(monkeypatch):
    # Create one dummy input and one dummy output gate.
    gate_in = DummyGate("In")
    inputs = DummyGatesGroup([gate_in])
    gate_out = DummyGate("Out")
    outputs = DummyGatesGroup([gate_out])
    initial_state = [(gate_out, 0.0)]
    sweeper = Sweeper(outputs=outputs, inputs=inputs, amplification=1.0, temperature="CT", device="Device1")
    # Override set_filename (fixing the typo if any) to do nothing.
    monkeypatch.setattr(sweeper, "set_filename", lambda: None)
    sweeper.is_save_file = False
    monkeypatch.setattr(time, "sleep", lambda x: None)
    monkeypatch.setattr("gate_manager.sweeper.tqdm", dummy_tqdm)
    # Run a very short time sweep.
    sweeper.sweepTime(measured_inputs=inputs, total_time=0.2, time_step=0.1,
                      initial_state=initial_state, comments="time_test", is_save_file=False)
    # Check that some current measurements were recorded.
    assert len(sweeper.currents) > 0
