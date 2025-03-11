import unittest
import tempfile
import os
import time
import matplotlib.pyplot as plt

# Import the Sweeper class from your module (adjust the import as needed)
from gate_manager.sweeper import Sweeper

# Define dummy classes to simulate Gate, GatesGroup, and Visualizer behavior.
class DummyLine:
    def __init__(self, label):
        self.label = label

class DummySource:
    def __init__(self, label):
        self.label = label

class DummyGate:
    def __init__(self, label, init_voltage=0.0):
        self.lines = [DummyLine(label)]
        self._voltage = init_voltage
        self.source = DummySource(label)

    def voltage(self, voltage=None, is_wait=True):
        # If called without an argument, return the current voltage (getter)
        if voltage is None:
            return self._voltage
        # Otherwise, set the voltage (setter)
        self._voltage = voltage

    def read_current(self, amplification):
        # Simulate a constant current (e.g., 0.1 uA)
        return 0.1

    def is_at_target_voltage(self, voltage):
        # For testing, assume the target voltage is always reached immediately
        return True

class DummyGatesGroup:
    def __init__(self, gates):
        self.gates = gates

    def voltage(self, voltage):
        for gate in self.gates:
            gate.voltage(voltage)

    def turn_off(self):
        for gate in self.gates:
            gate.voltage(0)

# Create a dummy Visualizer to override the real one in the sweeper module.
class DummyVisualizer:
    def viz2D(self, filename):
        # For testing, do nothing
        pass

# Override the Visualizer in the sweeper module with our dummy version.
import gate_manager.sweeper as sweeper
sweeper.Visualizer = DummyVisualizer

class TestSweeper(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory to run tests in isolation
        self.test_dir = tempfile.TemporaryDirectory()
        self.original_dir = os.getcwd()
        os.chdir(self.test_dir.name)
        
        # Patch time.sleep to avoid delays in tests
        self.original_sleep = time.sleep
        time.sleep = lambda s: None
        
        # Patch plt.pause to avoid waiting during interactive plotting
        self.original_pause = plt.pause
        plt.pause = lambda s: None

    def tearDown(self):
        # Restore patched functions and return to the original directory
        time.sleep = self.original_sleep
        plt.pause = self.original_pause
        os.chdir(self.original_dir)
        self.test_dir.cleanup()

    def test_sweep1D(self):
        # Create dummy output and measured input gates
        output_gate = DummyGate("Output1")
        measured_gate = DummyGate("Measured1")
        outputs_group = DummyGatesGroup([output_gate])
        measured_group = DummyGatesGroup([measured_gate])

        # Create a Sweeper instance with the dummy groups
        sweeper_obj = Sweeper(
            outputs=outputs_group,
            inputs=measured_group,
            amplification=1.0,
            temperature="300K",
            device="TestDevice"
        )

        # Define an initial state: a list of tuples (gate, initial_voltage)
        initial_state = [(output_gate, 0.0)]

        # Run the 1D voltage sweep with a minimal range for testing
        sweeper_obj.sweep1D(
            swept_outputs=outputs_group,
            measured_inputs=measured_group,
            start_voltage=0.0,
            end_voltage=0.1,
            step=0.1,
            initial_state=initial_state,
            voltage_unit='V',
            current_unit='uA',
            comments="test1D",
            ax2=None,
            is_2d_sweep=False
        )

        # Check that the log and figure files were created
        self.assertTrue(os.path.exists(f"{sweeper_obj.filename}.txt"))
        self.assertTrue(os.path.exists(f"{sweeper_obj.filename}.png"))

    def test_sweep2D(self):
        # Create dummy gates for X-sweep, Y-sweep, and measured input
        x_gate = DummyGate("X_Output")
        y_gate = DummyGate("Y_Output")
        measured_gate = DummyGate("Measured")
        X_group = DummyGatesGroup([x_gate])
        Y_group = DummyGatesGroup([y_gate])
        measured_group = DummyGatesGroup([measured_gate])

        # Create a Sweeper instance
        sweeper_obj = Sweeper(
            outputs=X_group,
            inputs=measured_group,
            amplification=1.0,
            temperature="300K",
            device="TestDevice"
        )

        # Define an initial state for the X-swept outputs
        initial_state = [(x_gate, 0.0)]

        # For testing the 2D sweep, override _log_params to bypass logging
        # (which would otherwise try to multiply None values).
        sweeper_obj._log_params = lambda sweep_type, status: None

        # Run the 2D sweep with minimal ranges (single step for each axis)
        sweeper_obj.sweep2D(
            X_swept_outputs=X_group,
            X_start_voltage=0.0,
            X_end_voltage=0.0,
            X_step=0.1,
            Y_swept_outputs=Y_group,
            Y_start_voltage=0.0,
            Y_end_voltage=0.0,
            Y_step=0.1,
            measured_inputs=measured_group,
            initial_state=initial_state,
            voltage_unit='V',
            current_unit='uA',
            comments="test2D"
        )

        # Verify that the data file was created (2D sweep writes data to self.filename.txt)
        self.assertTrue(os.path.exists(f"{sweeper_obj.filename}.txt"))

    def test_sweepTime(self):
        # Create dummy gates for outputs and measurement
        measured_gate = DummyGate("Measured")
        output_gate = DummyGate("Output1")
        outputs_group = DummyGatesGroup([output_gate])
        measured_group = DummyGatesGroup([measured_gate])

        # Create a Sweeper instance
        sweeper_obj = Sweeper(
            outputs=outputs_group,
            inputs=measured_group,
            amplification=1.0,
            temperature="300K",
            device="TestDevice"
        )

        # Define initial state for outputs
        initial_state = [(output_gate, 0.0)]

        # Run the time sweep with a very short duration for testing
        sweeper_obj.sweepTime(
            measured_inputs=measured_group,
            total_time=0.2,   # seconds
            time_step=0.1,    # seconds
            initial_state=initial_state,
            comments="testTime"
        )

        # Check that the log and figure files were created
        self.assertTrue(os.path.exists(f"{sweeper_obj.filename}.txt"))
        self.assertTrue(os.path.exists(f"{sweeper_obj.filename}.png"))

if __name__ == '__main__':
    unittest.main()
