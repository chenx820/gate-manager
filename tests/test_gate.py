# tests/test_gate.py

import pytest
from decimal import Decimal
import time

# Adjust the import path as needed.
from gate_manager.gate import Gate, GatesGroup

#########################
# Mock / Dummy Classes  #
#########################

class MockNanonis:
    """
    A mock for the Nanonis hardware interface.
    It provides methods that Gate expects: UserOut_ValSet, Signals_ValsGet, Signals_ValGet.
    """
    def __init__(self):
        self.set_voltages = {}
        self.read_voltages = {}
        self.read_currents = {}

    def UserOut_ValSet(self, write_index, voltage):
        # Store the voltage as passed (a Decimal)
        self.set_voltages[write_index] = voltage

    def Signals_ValsGet(self, read_indices, _bool):
        """
        Return a nested list so that:
            result[2][1][0][0] == voltage
        We use the first (and only) read_index.
        """
        read_index = read_indices[0]
        voltage = self.read_voltages.get(read_index, Decimal('0.0'))
        return [None, None, [None, [[voltage]]]]  # Note the nesting: [None, [[voltage]]] => [2][1][0][0]

    def Signals_ValGet(self, read_index, _bool):
        """
        Return a nested list so that:
            result[2][0] == current
        """
        current = self.read_currents.get(read_index, 0.0)
        return [None, None, [current, None]]


class MockNanonisSource:
    """
    A mock for the NanonisSource object that Gate expects.
    """
    def __init__(self, write_index=0, read_index=1, nanonis=None):
        self.write_index = write_index
        self.read_index = read_index
        self.nanonisInstance = nanonis


class MockSemiqonLine:
    """
    A mock for the SemiqonLine object, which just has a label.
    """
    def __init__(self, label):
        self.label = label


#########################
#       Fixtures        #
#########################

@pytest.fixture
def mock_nanonis():
    """Provides a fresh MockNanonis instance for each test."""
    return MockNanonis()

@pytest.fixture
def mock_source(mock_nanonis):
    """
    Provides a mock NanonisSource with:
      - write_index=0
      - read_index=1
      - nanonisInstance=mock_nanonis
    """
    return MockNanonisSource(write_index=0, read_index=1, nanonis=mock_nanonis)

@pytest.fixture
def mock_line():
    """Provides a mock SemiqonLine with a test label."""
    return MockSemiqonLine("TestLine")

@pytest.fixture
def gate(mock_source, mock_line):
    """
    Provides a Gate instance with:
      - source=mock_source
      - lines=[mock_line]
    """
    return Gate(source=mock_source, lines=[mock_line])

#########################
#     Gate Tests        #
#########################

def test_gate_init(gate, mock_line):
    """Test that the Gate is initialized with the correct label and no initial voltage."""
    assert gate.label == "TestLine"
    assert gate._voltage is None

def test_verify_in_range(gate):
    """Test that verify() does not raise an error for an in-range voltage."""
    gate.verify(2.0)  # Should not raise

def test_verify_out_of_range(gate):
    """Test that verify() raises ValueError for out-of-range voltages."""
    with pytest.raises(ValueError):
        gate.verify(5.0)

def test_set_volt_success(gate, mock_nanonis):
    """Test set_volt() for a valid write_index.
    
    Pass a Decimal so that conversion is exact.
    """
    gate.set_volt(Decimal("1.23"))
    assert mock_nanonis.set_voltages[gate.source.write_index] == Decimal("1.23")

def test_set_volt_no_write_index(mock_line, mock_nanonis):
    """Test set_volt() raises ValueError if write_index is None."""
    source = MockNanonisSource(write_index=None, read_index=1, nanonis=mock_nanonis)
    test_gate = Gate(source=source, lines=[mock_line])
    with pytest.raises(ValueError):
        test_gate.set_volt(Decimal("1.0"))

def test_get_volt(gate, mock_nanonis):
    """
    Test get_volt() reads the correct voltage from mock_nanonis.
    """
    mock_nanonis.read_voltages[gate.source.read_index] = Decimal("1.23")
    voltage = gate.get_volt()
    assert voltage == Decimal("1.23")
    assert gate._voltage == Decimal("1.23")

def test_voltage_get_only(gate, mock_nanonis):
    """Test that gate.voltage() returns the current voltage when no target is provided."""
    mock_nanonis.read_voltages[gate.source.read_index] = Decimal("2.34")
    voltage = gate.voltage()  # No target provided; should read current voltage.
    assert voltage == Decimal("2.34")

def test_voltage_set_and_wait(gate, mock_nanonis):
    """
    Test setting the voltage with is_wait=True.
    We'll simulate the voltage reaching the target.
    """
    # Initialize the read voltage at 0
    mock_nanonis.read_voltages[gate.source.read_index] = Decimal("0.0")

    def simulate_voltage_reach():
        mock_nanonis.read_voltages[gate.source.read_index] = Decimal("1.0")

    original_sleep = time.sleep

    def instant_sleep(_):
        simulate_voltage_reach()
        return original_sleep(0.0001)

    time.sleep = instant_sleep
    try:
        gate.voltage(Decimal("1.0"), is_wait=True)
        # Check that the final read voltage is 1.0
        assert gate._voltage == Decimal("1.0")
        # Also verify that the voltage was set in the mock
        assert mock_nanonis.set_voltages[gate.source.write_index] == Decimal("1.0")
    finally:
        time.sleep = original_sleep

def test_turn_off(gate, mock_nanonis):
    """Test that turn_off() sets voltage to 0.0."""
    gate.turn_off(is_wait=False)
    assert mock_nanonis.set_voltages[gate.source.write_index] == Decimal("0.0")

def test_is_at_target_voltage_true(gate, mock_nanonis):
    """Test is_at_target_voltage() returns True when the read voltage matches the target within tolerance."""
    mock_nanonis.read_voltages[gate.source.read_index] = Decimal("1.000001")
    assert gate.is_at_target_voltage(Decimal("1.0"), tolerance=Decimal("1e-5"))

def test_is_at_target_voltage_false(gate, mock_nanonis):
    """Test is_at_target_voltage() returns False when the voltage is out of tolerance."""
    mock_nanonis.read_voltages[gate.source.read_index] = Decimal("1.01")
    assert not gate.is_at_target_voltage(Decimal("1.0"), tolerance=Decimal("1e-6"))

def test_read_current(gate, mock_nanonis):
    """
    Test read_current() uses Signals_ValGet() to get the current and adjusts it by the amplification factor.
    """
    # Suppose the raw current reading is 5e-7 (0.5 µA)
    mock_nanonis.read_currents[gate.source.read_index] = 5e-7
    current = gate.read_current(amplification=-1e6)
    # Calculation: 5e-7 * 10^6 / -1e6 = -5e-7
    assert abs(current - Decimal("-5E-7")) < Decimal("1E-10")

#########################
#  GatesGroup Tests     #
#########################

@pytest.fixture
def gate_group(mock_nanonis):
    """
    Provides a GatesGroup with two Gate objects having distinct indices.
    """
    line1 = MockSemiqonLine("Line1")
    line2 = MockSemiqonLine("Line2")

    source1 = MockNanonisSource(write_index=0, read_index=10, nanonis=mock_nanonis)
    source2 = MockNanonisSource(write_index=1, read_index=11, nanonis=mock_nanonis)

    gate1 = Gate(source=source1, lines=[line1])
    gate2 = Gate(source=source2, lines=[line2])

    return GatesGroup([gate1, gate2])

def test_gates_group_set_volt(gate_group, mock_nanonis):
    """Test that set_volt() sets all gates in the group to the same voltage."""
    gate_group.set_volt(Decimal("2.0"))
    assert mock_nanonis.set_voltages[0] == Decimal("2.0")
    assert mock_nanonis.set_voltages[1] == Decimal("2.0")

def test_gates_group_voltage_wait(gate_group, mock_nanonis):
    """
    Test that voltage() sets all gates and waits until they reach the target voltage.
    """
    # Initialize read voltages to 0
    mock_nanonis.read_voltages[10] = Decimal("0.0")
    mock_nanonis.read_voltages[11] = Decimal("0.0")

    def simulate_voltage_reach():
        mock_nanonis.read_voltages[10] = Decimal("1.0")
        mock_nanonis.read_voltages[11] = Decimal("1.0")

    original_sleep = time.sleep

    def instant_sleep(_):
        simulate_voltage_reach()
        return original_sleep(0.0001)

    time.sleep = instant_sleep
    try:
        gate_group.voltage(Decimal("1.0"), is_wait=True)
        for gate_obj in gate_group.gates:
            assert gate_obj._voltage == Decimal("1.0")
    finally:
        time.sleep = original_sleep

def test_gates_group_turn_off(gate_group, mock_nanonis):
    """Test that turn_off() sets all gates in the group to 0.0."""
    gate_group.turn_off(is_wait=False)
    assert mock_nanonis.set_voltages[0] == Decimal("0.0")
    assert mock_nanonis.set_voltages[1] == Decimal("0.0")
