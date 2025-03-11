import pytest
import time
from gate_manager.gate import Gate, GatesGroup
from gate_manager.connection import NanonisSource, SemiqonLine

# Dummy Nanonis class to simulate the required methods.
class DummyNanonis:
    def __init__(self):
        # Dictionary to store voltage values keyed by index.
        self.voltages = {}

    def UserOut_ValSet(self, write_index, voltage):
        """
        Simulate setting the voltage on the output.
        For simplicity, assume that write_index and read_index are identical.
        """
        self.voltages[write_index] = voltage

    def Signals_ValsGet(self, read_indices, flag):
        """
        Return a nested structure such that [2][1][0][0] equals the voltage for the given read index.
        """
        voltage = self.voltages.get(read_indices[0], 0)
        # Construct a nested list: index 2 -> list with a second element,
        # whose first element is a list with first element being the voltage.
        return [None, None, [None, [[voltage]]]]

    def Signals_ValGet(self, read_index, flag):
        """
        Return a structure such that [2][0] equals the voltage for the given read index.
        """
        voltage = self.voltages.get(read_index, 0)
        return [None, None, [voltage]]


# -----------------------------
# Fixtures for Dummy Nanonis and Sources
# -----------------------------

@pytest.fixture
def dummy_nanonis():
    """Fixture for a dummy Nanonis instance."""
    return DummyNanonis()


@pytest.fixture
def dummy_source(dummy_nanonis):
    """
    Fixture for a dummy NanonisSource with matching read and write indices.
    Using index 1 for simplicity.
    """
    return NanonisSource(label="Test Source", read_index=1, write_index=1, nanonisInstance=dummy_nanonis)


@pytest.fixture
def gate(dummy_source):
    """
    Fixture for a Gate instance with a dummy source and a single SemiqonLine.
    """
    line = SemiqonLine(label="Line1")
    return Gate(source=dummy_source, lines=[line])


# -----------------------------
# Tests for Gate class
# -----------------------------

def test_gate_verify_valid(gate):
    """
    Test that verify() does not raise an exception for a valid voltage.
    """
    try:
        gate.verify(0.0)
    except Exception as e:
        pytest.fail(f"verify() raised an exception for a valid voltage: {e}")


def test_gate_verify_invalid(gate):
    """
    Test that verify() raises a ValueError when the voltage is out of range.
    """
    with pytest.raises(ValueError):
        gate.verify(3.0)  # Out of allowed range (-2.5V to 2.5V)


def test_gate_set_volt_valid(gate, dummy_nanonis):
    """
    Test that set_volt() correctly sets the voltage using the dummy Nanonis instance.
    """
    test_voltage = 1.5
    gate.set_volt(test_voltage)
    # Since the source's read_index equals write_index, the dummy voltage should be updated.
    assert dummy_nanonis.voltages.get(gate.source.write_index) == test_voltage


def test_gate_set_volt_read_only(dummy_nanonis):
    """
    Test that set_volt() raises a ValueError when the source is read-only (write_index is None).
    """
    source = NanonisSource(label="ReadOnly Source", read_index=2, write_index=None, nanonisInstance=dummy_nanonis)
    gate_obj = Gate(source=source, lines=[SemiqonLine(label="Line1")])
    with pytest.raises(ValueError):
        gate_obj.set_volt(1.0)


def test_gate_get_volt(gate, dummy_nanonis):
    """
    Test that get_volt() returns the correct voltage from the dummy Nanonis instance.
    """
    dummy_nanonis.voltages[gate.source.read_index] = 2.0
    voltage = gate.get_volt()
    assert voltage == 2.0


def test_gate_voltage_get(gate, dummy_nanonis):
    """
    Test that voltage() without target_voltage returns the current voltage.
    """
    dummy_nanonis.voltages[gate.source.read_index] = 1.2
    result = gate.voltage()
    assert result == 1.2


def test_gate_voltage_set(gate, dummy_nanonis):
    """
    Test that voltage() with a target_voltage sets the voltage and waits until it is reached.
    """
    test_voltage = -1.5
    gate.voltage(test_voltage, is_wait=True)
    # Check that the dummy Nanonis voltage has been updated.
    assert dummy_nanonis.voltages.get(gate.source.write_index) == test_voltage


def test_gate_turn_off(gate, dummy_nanonis):
    """
    Test that turn_off() sets the gate voltage to zero.
    """
    dummy_nanonis.voltages[gate.source.write_index] = 1.0
    gate.turn_off(is_wait=True)
    assert dummy_nanonis.voltages.get(gate.source.write_index) == 0.0


def test_gate_is_at_target_voltage(gate, dummy_nanonis):
    """
    Test that is_at_target_voltage() correctly compares the current voltage with the target.
    """
    dummy_nanonis.voltages[gate.source.read_index] = 0.5
    assert gate.is_at_target_voltage(0.5)
    dummy_nanonis.voltages[gate.source.read_index] = 0.5 + 1e-4
    assert not gate.is_at_target_voltage(0.5)


def test_gate_read_current(gate, dummy_nanonis):
    """
    Test that read_current() returns the expected current reading.
    """
    dummy_nanonis.voltages[gate.source.read_index] = 2.0
    # With amplification = -10**6, expected current = 2.0 * 10**6 / (-10**6) = -2.0
    current = gate.read_current(amplification=-10**6)
    assert current == -2.0


# -----------------------------
# Fixtures for GatesGroup tests
# -----------------------------

@pytest.fixture
def gate1(dummy_nanonis):
    """
    Fixture for a Gate instance (Gate 1) with index 1.
    """
    source = NanonisSource(label="Gate1 Source", read_index=1, write_index=1, nanonisInstance=dummy_nanonis)
    line = SemiqonLine(label="Line1")
    return Gate(source=source, lines=[line])


@pytest.fixture
def gate2(dummy_nanonis):
    """
    Fixture for a Gate instance (Gate 2) with index 2.
    """
    source = NanonisSource(label="Gate2 Source", read_index=2, write_index=2, nanonisInstance=dummy_nanonis)
    line = SemiqonLine(label="Line2")
    return Gate(source=source, lines=[line])


@pytest.fixture
def gates_group(gate1, gate2):
    """
    Fixture for a GatesGroup instance containing two Gate objects.
    """
    return GatesGroup(gates=[gate1, gate2])


# -----------------------------
# Tests for GatesGroup class
# -----------------------------

def test_gates_group_set_volt(gates_group, dummy_nanonis):
    """
    Test that set_volt() sets the voltage for all gates in the group.
    """
    target_voltage = 1.0
    gates_group.set_volt(target_voltage)
    for gate_obj in gates_group.gates:
        assert dummy_nanonis.voltages.get(gate_obj.source.write_index) == target_voltage


def test_gates_group_voltage(gates_group, dummy_nanonis):
    """
    Test that voltage() sets the voltage for all gates in the group and waits until the target is reached.
    """
    target_voltage = -1.0
    gates_group.voltage(target_voltage, is_wait=True)
    for gate_obj in gates_group.gates:
        assert dummy_nanonis.voltages.get(gate_obj.source.write_index) == target_voltage


def test_gates_group_turn_off(gates_group, dummy_nanonis):
    """
    Test that turn_off() sets the voltage of all gates in the group to zero.
    """
    # Set a nonzero voltage for each gate.
    for gate_obj in gates_group.gates:
        dummy_nanonis.voltages[gate_obj.source.write_index] = 1.5
    gates_group.turn_off(is_wait=True)
    for gate_obj in gates_group.gates:
        assert dummy_nanonis.voltages.get(gate_obj.source.write_index) == 0.0
