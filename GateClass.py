# -*- coding: utf-8 -*-
"""
Gate class to interface with the Nanonis system for reading and setting voltages.

Created on Tue Oct 22 16:08:06 2024
@author: Chen Huang <chen.huang23@imperial.ac.uk>
"""
from nanonis_spm import Nanonis
from decimal import Decimal
import time


class Gate:
    """
    A class representing a gate used in experiments interfacing with the Nanonis system.

    Attributes:
        name (str): The name of the gate.
        label (str): A label identifying the gate.
        read_index (int): The index used to read voltage from the gate.
        write_index (int, optional): The index used to write voltage to the gate (None if not writable).
        nanonisInstance (Nanonis): An instance of the Nanonis class for communication with the device.
    """

    def __init__(self, name: str = None, label: str = None, read_index: int = None, write_index: int = None,
                 nanonisInstance: Nanonis = None):
        """
        Initializes the Gate with its name, label, read/write indices, and the Nanonis instance.

        Args:
            name (str, optional): The name of the gate.
            label (str, optional): The label for the gate.
            read_index (int, optional): The index for reading the voltage.
            write_index (int, optional): The index for writing the voltage (if applicable).
            nanonisInstance (Nanonis): The Nanonis instance for hardware communication.
        """
        self.name = name
        self.label = label
        self.read_index = read_index
        self.write_index = write_index
        self.nanonisInstance = nanonisInstance
        self._voltage = self.get_volt()

    def set_volt(self, target_voltage: float or Decimal) -> None:
        """
        Sets the voltage for the gate.

        Args:
            target_voltage (float or Decimal): The voltage value to set.

        Raises:
            ValueError: If the write_index is None, indicating the gate cannot set voltage.
        """
        if self.write_index is None:
            raise ValueError(
                f"'{self.name}' cannot set voltage because write_index is not defined.")
        else:
            self.nanonisInstance.UserOut_ValSet(self.write_index, Decimal(target_voltage))

    def get_volt(self) -> Decimal:
        """
        Retrieves the current voltage from the gate.

        Returns:
            Decimal: The current voltage.
        """
        self._voltage = Decimal(self.nanonisInstance.Signals_ValsGet([self.read_index], True)[2][1][0][0])
        return self._voltage

    def voltage(self, target_voltage: float or Decimal = None, wait: bool = True) -> Decimal:
        """
        Gets or sets the voltage. If no value is provided, it reads the current voltage.

        Args:
            target_voltage (float or Decimal, optional): The voltage value to set. If None, the current voltage is retrieved.
            wait (bool, option): Whether to wait for the current to reach the target voltage.

        Returns:
            Decimal: The set or retrieved voltage.
        """
        if target_voltage is None:
            self.get_volt()
            return self._voltage
        else:
            self.set_volt(target_voltage)
            if wait:
                print(f"[INFO] Ramping {self.label} to {target_voltage} [V]. ")
                while True:
                    if self.is_at_target_voltage(target_voltage):
                        break
                    time.sleep(0.01)
                print(f"[INFO] {self.label} is at {target_voltage} [V]. ")
            return Decimal(target_voltage)

    def turn_off(self, wait: bool = True):
        self.voltage(0.0, wait)

    def is_at_target_voltage(self, target_voltage: float or Decimal, tolerance: float or Decimal = 1e-6) -> bool:
        """
        Check if the current voltage is within the specified tolerance of the target voltage.

        Args:
            target_voltage (float or Decimal): the target voltage value.
            tolerance (float): The allowed deviation from the target value.

        Returns:
            bool: True if the voltage is within the specified tolerance of the target voltage.
        """
        self.get_volt()
        return abs(self._voltage - Decimal(target_voltage)) < Decimal(tolerance)

    def read_current(self, amplifier: float = -1) -> Decimal:
        """
        Reads the current from the gate.

        Args:
            amplifier (float, optional): The amplifier setting, default is -1.

        Returns:
            float: The current value adjusted by the amplifier.
        """
        return Decimal(self.nanonisInstance.Signals_ValGet(self.read_index, True)[2][0] * amplifier)

    def set_label(self, label: str) -> None:
        """
        Sets the label for the gate.

        Args:
            label (str): The new label for the gate.
        """
        self.label = label

    def set_name(self, name: str) -> None:
        """
        Sets the name for the gate.

        Args:
            name (str): The new name for the gate.
        """
        self.name = name
