import unittest
from unittest.mock import MagicMock, patch
from decimal import Decimal
from gate_manager.gate import Gate, GatesGroup
from gate_manager.connection import SemiqonLine, NanonisSource

class TestGate(unittest.TestCase):
    def setUp(self):
        """Setup the test environment."""
        # Mock the NanonisSource and SemiqonLine
        self.mock_nanonis = MagicMock()
        self.mock_nanonisInstance = MagicMock()
        self.mock_nanonis.Signals_ValsGet.return_value = [[0], [0], [[0, [0, 0]]]]
        
        self.mock_source = MagicMock(spec=NanonisSource)
        self.mock_source.nanonisInstance = self.mock_nanonisInstance
        self.mock_source.write_index = 1
        self.mock_source.read_index = 2
        
        # Create a list of SemiqonLine objects and mock them
        self.mock_lines = [MagicMock(spec=SemiqonLine) for _ in range(2)]
        for line in self.mock_lines:
            line.label = "line_label"
        
        self.gate = Gate(source=self.mock_source, lines=self.mock_lines)

    def test_set_voltage(self):
        """Test setting voltage."""
        self.gate.set_volt(1.5)
        self.mock_nanonisInstance.UserOut_ValSet.assert_called_with(1, Decimal(1.5))

    def test_get_voltage(self):
        """Test getting the voltage."""
        self.mock_nanonisInstance.Signals_ValsGet.return_value = [[0], [0], [[0, [0, 1.5]]]]
        voltage = self.gate.get_volt()
        self.assertEqual(voltage, Decimal(1.5))

    def test_verify_voltage_valid(self):
        """Test that valid voltages pass the verification."""
        try:
            self.gate.verify(1.0)  # Valid voltage within the range
        except ValueError:
            self.fail("verify() raised ValueError unexpectedly")

    def test_verify_voltage_invalid(self):
        """Test that invalid voltages raise a ValueError."""
        with self.assertRaises(ValueError):
            self.gate.verify(5.0)  # Invalid voltage beyond the range

    def test_turn_off(self):
        """Test turning off the gate (voltage = 0)."""
        self.gate.turn_off()
        self.mock_nanonisInstance.UserOut_ValSet.assert_called_with(1, Decimal(0.0))

    def test_is_at_target_voltage(self):
        """Test checking if the voltage is within tolerance."""
        self.mock_nanonisInstance.Signals_ValsGet.return_value = [[0], [0], [[0, [0, 1.5]]]]
        self.gate._voltage = Decimal(1.5)
        self.assertTrue(self.gate.is_at_target_voltage(1.5))
        self.assertFalse(self.gate.is_at_target_voltage(1.0))

    def test_read_current(self):
        """Test reading the current from the gate."""
        self.mock_nanonisInstance.Signals_ValGet.return_value = [[0], [0], [1]]
        current = self.gate.read_current()
        self.assertEqual(current, Decimal(1e-6))  # Assuming amplification factor is -10**6


class TestGatesGroup(unittest.TestCase):
    def setUp(self):
        """Setup the test environment for GatesGroup."""
        self.mock_gate_1 = MagicMock(spec=Gate)
        self.mock_gate_2 = MagicMock(spec=Gate)
        self.gates_group = GatesGroup(gates=[self.mock_gate_1, self.mock_gate_2])

    def test_set_volt_group(self):
        """Test setting the voltage for the entire group."""
        self.gates_group.set_volt(2.0)
        self.mock_gate_1.set_volt.assert_called_with(2.0)
        self.mock_gate_2.set_volt.assert_called_with(2.0)

    def test_turn_off_group(self):
        """Test turning off the entire group of gates."""
        self.gates_group.turn_off()
        self.mock_gate_1.turn_off.assert_called_with(is_wait=True)
        self.mock_gate_2.turn_off.assert_called_with(is_wait=True)

    def test_voltage_group(self):
        """Test setting and waiting for voltage for the entire group."""
        self.gates_group.voltage(2.0)
        self.mock_gate_1.voltage.assert_called_with(2.0, False)
        self.mock_gate_2.voltage.assert_called_with(2.0, False)


if __name__ == "__main__":
    unittest.main()