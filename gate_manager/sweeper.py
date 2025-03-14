# -*- coding: utf-8 -*-
"""
Sweeper Class for Conducting Voltage Sweeps with the Nanonis System.

This module provides the Sweeper class to perform 1D and 2D voltage sweeps
across a set of gates using the Nanonis system. It logs measurement data and
generates animated plots for analysis. The class enables precise control of sweep
parameters and records experimental metadata.

Classes:
    Sweeper: Conducts voltage sweeps on specified gates, logs results, and
             generates plots for analysis.
             
Created on Wed Nov 06 10:46:06 2024
@author:
Chen Huang <chen.huang23@imperial.ac.uk>
"""

from datetime import datetime, date
import math
import time
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .gate import GatesGroup, Gate
from .visualizer import Visualizer


class Sweeper:
    """
    Sweeper class to perform and log voltage sweeps on defined gates.
    """

    def __init__(self, 
                 outputs: GatesGroup = None, 
                 inputs: GatesGroup = None, 
                 temperature: str = None, 
                 device: str = None) -> None:
        self.outputs = outputs
        self.inputs = inputs
        self.temperature = temperature
        self.device = device

        # Labels and file metadata
        self.x_label = None
        self.y_label = None
        self.z_label = None
        self.comments = None
        self.filename = None

        # Sweep configuration
        self.X_start_voltage = None
        self.X_end_voltage = None
        self.X_step = None
        
        self.Y_start_voltage = None
        self.Y_end_voltage = None
        self.Y_step = None
        
        self.total_time = None
        self.time_step = None

        # Measurement data
        self.X_voltage = None
        self.Y_voltage = None
        
        self.X_voltages = []
        self.currents = []
        self.is_2d_sweep = False
        
        # Units
        self.voltage_unit = 'V'
        self.current_unit = 'uA'
        self.voltage_scale = 1
        self.current_scale = 1
        
        self.init_folders()
        
    def init_folders(self) -> None:
        """Create folders for saving data and figures."""
        # Choose the currents folder
        current_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(current_dir)

        # Define the folder names to check
        folders = ['figures', 'data']

        # Check if each folder exists and create it if it doesn't
        for folder in folders:
            if not os.path.exists(folder):
                os.makedirs(folder)
        
    def _set_units(self) -> None:
        """Set voltage and current units."""
        unit_map = {'V': 1, 'mV': 1e3, 'uV': 1e6}
        self.voltage_scale = unit_map.get(self.voltage_unit, 1)
        
        unit_map = {'mA': 1e-3, 'uA': 1, 'nA': 1e3, 'pA': 1e6}
        self.current_scale = unit_map.get(self.current_unit, 1)
        
    def _convert_units(self, voltage_pack: list[float, str]) -> float:
        """Convert voltage unit to V.
        
        Args:
            voltage_pack (list): [value, unit], e.g. [1.0, 'mV'].
        """
        voltage, unit = voltage_pack
        unit_map = {'V': 1e0, 'mV': 1e-3, 'uV': 1e-6, 'nV': 1e-9}
        return voltage * unit_map.get(unit, 1)
    
    def convert_si_value(self, value, unit):
        """
        Convert a given numerical value and its unit to an appropriate SI prefixed representation,
        so that the resulting number falls within the range [1, 1000) (or is 0).
        
        Args:
            value (float or int): The numerical value 
            unit (str): Unit string, e.g., "V", "mV", "kV", etc. (assuming the prefix is a single character)
        
        Returns:
            str e.g., 100.000 [mV]
        """
        # Define multipliers corresponding to SI prefixes (includes common prefixes)
        prefixes = {
        'Y': 1e24, 'Z': 1e21, 'E': 1e18, 'P': 1e15, 'T': 1e12,
        'G': 1e9, 'M': 1e6, 'k': 1e3, '': 1, 'm': 1e-3, 'μ': 1e-6, 'u': 1e-6,
        'n': 1e-9, 'p': 1e-12, 'f': 1e-15, 'a': 1e-18, 'z': 1e-21, 'y': 1e-24
        }
        # Try to extract the prefix from the unit (assuming the prefix is a single character
        # followed by the base unit)
        if len(unit) > 1 and unit[0] in prefixes and unit[1].isalpha():
            prefix = unit[0]
            base_unit = unit[1:]
        else:
            prefix = ''
            base_unit = unit
            
        # Convert the input value to the value in the base unit
        base_value = value * prefixes[prefix]
        
        # Define mapping from exponent (in multiples of 3) to SI prefixes
        si_prefixes = {
            -24: 'y', -21: 'z', -18: 'a', -15: 'f', -12: 'p',
            -9: 'n', -6: 'u', -3: 'm', 0: '', 3: 'k',
            6: 'M', 9: 'G', 12: 'T', 15: 'P', 18: 'E',
            21: 'Z', 24: 'Y'
            }
        
        # If the value is 0, return immediately
        if base_value == 0:
            return f"{0:>7.3f} [{base_unit}]"
        
        # Calculate the order of magnitude of the base value
        exponent = int(math.floor(math.log10(abs(base_value))))
        # Round down the exponent to the nearest multiple of 3
        exponent3 = (exponent // 3) * 3
        # Ensure exponent3 is within the available range of si_prefixes
        min_exp = min(si_prefixes.keys())
        max_exp = max(si_prefixes.keys())
        exponent3 = max(min_exp, min(max_exp, exponent3))
        
        # Calculate the converted value and corresponding SI prefixed unit
        new_value = base_value / (10**exponent3)
        new_unit = si_prefixes[exponent3] + base_unit
        return f"{new_value:>7.3f} [{new_unit}]"

    def _set_gates_group_label(self, gates_group: GatesGroup) -> str:
        """Generate a label by combining the labels from all lines in a group of gates."""
        return " & ".join(line.label for gate in gates_group.gates for line in gate.lines)

    def _set_gate_label(self, gate: Gate) -> str:
        """Generate a label for a single gate by combining its line labels."""
        return " & ".join(line.label for line in gate.lines)

    def _set_filename(self, prefix: str) -> None:
        """Generate a unique filename for saving data."""
        if prefix == '1D':
            base_filename = f"{date.today().strftime('%Y%m%d')}_{self.temperature}_[{self.z_label}]_vs_[{self.x_label}]"
        elif prefix == '2D':
            base_filename = f"{date.today().strftime('%Y%m%d')}_{self.temperature}_[{self.z_label}]_vs_[{self.x_label}]_[{self.y_label}]"
        elif prefix == 'time':
            base_filename = f"{date.today().strftime('%Y%m%d')}_{self.temperature}_[{self.y_label}]_vs_time"
        else:
            raise ValueError("Invalid prefix for filename.")
        if self.comments:
            base_filename += f"_{self.comments}"
        self.filename = self._get_unique_filename(base_filename)

    def _get_unique_filename(self, base_filename: str) -> str:
        """Ensure unique filenames to prevent overwriting."""
        filepath = os.path.join(os.getcwd(), f"data/{base_filename}")

        counter = 1
        while os.path.isfile(f"{filepath}_run{counter}.txt"):
            counter += 1
        return f"{base_filename}_run{counter}"
            

    def _log_params(self, sweep_type: str = 'voltage', status: str = 'start') -> None:
        """
        Log sweep parameters and experimental metadata to a file.

        Args:
            sweep_type (str): Type of sweep ('voltage', 'time', etc.) to log specific parameters.
            status (str): 'start' or 'end' of the run.
        """
        if status == 'start':
            self.log_filename = "log"
            if self.comments:
                self.log_filename += f"_{self.comments}"
            with open(f"{self.log_filename}.txt", 'a') as file:
                self.start_time = datetime.now()
                file.write(
                    f"--------/// Run started at {self.start_time.strftime('%Y-%m-%d %H:%M:%S')} ///--------\n")
                file.write(f"{'Filename:':<16} {self.filename}.txt \n")
                file.write(f"{'Device:':<16} {self.device} \n")
                file.write(f"{'Measured Input:':<16} {self.z_label} \n")
                file.write("\n")
                file.write(f"{'X Swept Gates:':<16} {self.x_label} \n")
                if sweep_type == 'voltage':
                    file.write(f"{'Start Voltage:':<16} {self.convert_si_value(self.X_start_voltage, 'V')} \n")
                    file.write(f"{'End Voltage:':<16} {self.convert_si_value(self.X_end_voltage, 'V')} \n")
                    file.write(f"{'Step Size:':<16} {self.convert_si_value(self.X_step, 'V')} \n")
                    file.write("\n")
                if self.is_2d_sweep:
                    file.write(f"{'Y Swept Gates:':<16} {self.y_label} \n")
                    file.write(f"{'Start Voltage:':<16} {self.convert_si_value(self.Y_start_voltage, 'V')} \n")
                    file.write(f"{'End Voltage:':<16} {self.convert_si_value(self.Y_end_voltage, 'V')} \n")
                    file.write(f"{'Step Size:':<16} {self.convert_si_value(self.Y_step, 'V')} \n")
                    file.write("\n")
                if sweep_type == 'time':
                    file.write(f"{'Total Time:':<16} {self.total_time:>16.2f} [s] \n")
                    file.write(f"{'Time Step:':<16} {self.time_step:>16.2f} [s] \n")
                    file.write("\n")
                file.write("Initial Voltages of all outputs before sweep: \n")
                for output_gate in self.outputs.gates:
                    voltage = output_gate.voltage()
                    file.write(
                        f"{' & '.join(line.label for line in output_gate.lines):<55} {self.convert_si_value(voltage, 'V')} \n")
                file.write("\n")
        if status == 'end':
            total_time_elapsed = datetime.now() - self.start_time
            hours, remainder = divmod(total_time_elapsed.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            with open(f"{self.log_filename}.txt", 'a') as file:
                file.write(f"{'Total Time:':<16} {int(hours)}h {int(minutes)}m {int(seconds)}s \n")
                file.write(
                    f"--------/// Run ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ///--------\n")
                file.write("\n")

    def sweep1D(self, 
                swept_outputs: GatesGroup, 
                measured_inputs: GatesGroup, 
                start_voltage: list[float, str], 
                end_voltage: list[float, str],
                step: list[float, str], 
                initial_state: list = None, 
                voltage_unit: str = 'V',
                current_unit: str = 'uA',
                comments: str = None, 
                ax2=None, 
                is_2d_sweep: bool = False,
                is_show: bool = True
                ) -> tuple:
        """
        Perform a 1D voltage sweep and generate an animated plot.

        Args:
            swept_outputs (GatesGroup): Group of output gates to sweep.
            measured_inputs (GatesGroup): Group of input gates for current measurement.
            start_voltage (list): Starting voltage as [value, unit].
            end_voltage (list): Ending voltage as [value, unit].
            step (list): Voltage increment for each step as [value, unit].
            initial_state (list): List of tuples (gate, init_voltage) where init_voltage is [value, unit].
            voltage_unit (str): Unit for voltage for display.
            current_unit (str): Unit for current for display.
            comments (str): Additional comments for logging.
            ax2: Optional axis for plotting if already provided.
            is_2d_sweep (bool): Flag indicating whether this sweep is part of a 2D sweep.

        Returns:
            tuple: (voltages, current_values) if is_2d_sweep is True, else None.
        """
        
        # Set sweep labels and units
        self.x_label = self._set_gates_group_label(swept_outputs)
        self.z_label = self._set_gates_group_label(measured_inputs)
        self.voltage_unit = voltage_unit
        self.current_unit = current_unit
        self.comments = comments
        self.ax2 = ax2
        self.is_2d_sweep = is_2d_sweep
        
        self._set_units()
        
        if not self.is_2d_sweep:
            self._set_filename('1D')

        self.X_start_voltage = self._convert_units(start_voltage)
        self.X_end_voltage = self._convert_units(end_voltage)
        self.X_step = self._convert_units(step)
        
        if self.X_step < 0:
            raise ValueError("Step size must be positive.")

        pbar = tqdm(total=len(initial_state)+len(swept_outputs.gates), desc="[INFO] Ramping voltage", ncols=80,
                    leave=True)
        
        # Set initial state for designated gates
        converted_init_state = []
        for gate, init_volt, init_unit in initial_state:
            converted_init_volt = self._convert_units([init_volt, init_unit])
            converted_init_state.append([gate, converted_init_volt])
            gate.voltage(converted_init_volt, is_wait=False)

        # Wait until all initial voltages stabilize
        while not all([gate.is_at_target_voltage(voltage) for gate, voltage in converted_init_state]):
            time.sleep(0.1)
        pbar.update(len(initial_state))

        # Set swept outputs to the starting voltage
        swept_outputs.voltage(self.X_start_voltage)
        pbar.update(len(swept_outputs.gates))
        pbar.close()
        time.sleep(0.1)

        # TO DO: If there is more than one measured input? 
        
        # Set up plotting
        if self.ax2 is None:
            plt.ion()
            fig, self.ax2 = plt.subplots(1, 1, figsize=(12, 7))
        else:
            self.ax2.clear()
            self.ax2.set_title(f"{self.y_label}: {self.convert_si_value(self.Y_voltage, 'V')}")
        self.ax2.set_xlabel(f"{self.x_label} [{self.voltage_unit}]")
        self.ax2.set_ylabel(f"{self.z_label} [{self.current_unit}]")
        
        
        self.currents = []
        self.X_voltage = self.X_start_voltage
        self.X_voltages = []

        # Log sweep parameters (for non-2D sweeps)
        if not self.is_2d_sweep:
            self._log_params(sweep_type='voltage', status='start')
            with open(f"data/{self.filename}.txt", 'a') as file:
                header = (f"{self.x_label} [{self.voltage_unit}]".rjust(16) + 
                          f"{self.z_label} [{self.current_unit}]".rjust(16))
                file.write(header + "\n")

        print(
            f"[INFO] Start sweeping {self.x_label} from {self.X_start_voltage*self.voltage_scale} " 
            f"[{self.voltage_unit}] to {self.X_end_voltage*self.voltage_scale} [{self.voltage_unit}].")
        
        self.lines, = self.ax2.plot([], [])
        total_steps = round(abs(self.X_end_voltage - self.X_start_voltage) / self.X_step + 1)
        pbar = tqdm(total=total_steps, desc="[INFO] Sweeping", ncols=80, leave=True) 
        frame = 0
        
        while True:
            swept_outputs.voltage(self.X_voltage)
            self.X_voltages.append(self.X_voltage * self.voltage_scale)
            
            # Read current from the first measured input (extend as needed)
            current = measured_inputs.gates[0].read_current() * self.current_scale
            self.currents.append(current)
            
            # Update plot limits and data
            self.ax2.set_xlim(
                min(self.X_voltages) - self.X_step * self.voltage_scale, 
                max(self.X_voltages) + self.X_step * self.voltage_scale
                )
            curr_min = min(self.currents)
            curr_max = max(self.currents)
            if curr_min == curr_max:
                curr_min -= 0.001
                curr_max += 0.001
            self.ax2.set_ylim(min(self.currents) - (curr_max - curr_min) / 4,
                              max(self.currents) + (curr_max - curr_min) / 4)
            self.lines.set_data(self.X_voltages, self.currents)

            plt.draw()
            plt.pause(0.01)
            frame += 1
            pbar.update(1)

            with open(f"data/{self.filename}.txt", 'a') as file:
                if self.is_2d_sweep:
                    file.write(f"{self.Y_voltage * self.voltage_scale:>16.4f} " 
                               f"{self.X_voltage * self.voltage_scale:>16.4f} "
                               f"{current:>16.8f} \n")
                else: 
                    file.write(f"{self.X_voltage * self.voltage_scale:>16.4f} "
                               f"{current:>16.8f} \n")
                    
            # Check if sweep is complete    
            if (self.X_start_voltage < self.X_end_voltage and self.X_voltage > self.X_end_voltage - 1e-6) or (
                    self.X_start_voltage > self.X_end_voltage and self.X_voltage < self.X_end_voltage + 1e-6):
                pbar.close()
                break
            self.X_voltage = self.X_start_voltage + frame * self.X_step \
                if self.X_start_voltage < self.X_end_voltage \
                else self.X_start_voltage - frame * self.X_step
        
        if self.is_2d_sweep:
            print("\n")
            return self.X_voltages, self.currents
        else:
            plt.ioff()
            plt.tight_layout()
            plt.savefig(f"figures/{self.filename}.png", dpi=300, bbox_inches='tight')
            if is_show:
                plt.show()
            else:
                plt.close()
            print("[INFO] Data collection complete and figure saved. \n")
            self._log_params(sweep_type='voltage', status='end')
            

    def sweep2D(self, 
                X_swept_outputs: GatesGroup, 
                X_start_voltage: list[float, str], 
                X_end_voltage: list[float, str], 
                X_step: list[float, str], 
                Y_swept_outputs: GatesGroup, 
                Y_start_voltage: list[float, str], 
                Y_end_voltage: list[float, str], 
                Y_step: list[float, str], 
                measured_inputs: GatesGroup, 
                initial_state: list, 
                voltage_unit: str = 'V',
                current_unit: str = 'uA',
                comments: str = None,
                is_show: bool = True):
        """
        Perform a 2D voltage sweep over two axes by sweeping one set of outputs for each voltage
        setting of another set.

        Args:
            X_swept_outputs (GatesGroup): Gates to sweep along the X axis.
            X_start_voltage (list): Starting voltage for X axis as [value, unit].
            X_end_voltage (list): Ending voltage for X axis as [value, unit].
            X_step (list): Voltage step for X axis as [value, unit].
            Y_swept_outputs (GatesGroup): Gates to sweep along the Y axis.
            Y_start_voltage (list): Starting voltage for Y axis as [value, unit].
            Y_end_voltage (list): Ending voltage for Y axis as [value, unit].
            Y_step (list): Voltage step for Y axis as [value, unit].
            measured_inputs (GatesGroup): Group of input gates for measurements.
            initial_state (list): List of tuples (gate, init_voltage) where init_voltage is [value, unit].
            voltage_unit (str): Voltage unit for display.
            current_unit (str): Current unit for display.
            comments (str): Additional comments for logging.
        """
        self.voltage_unit = voltage_unit
        self.current_unit = current_unit
        self.is_2d_sweep = True
        
        self._set_units()
        
        # Prepare parameters for the 1D sweep call
        params = {
            # here we use the variable name for the gate which is okay
            'swept_outputs': X_swept_outputs,
            'start_voltage': X_start_voltage,
            'end_voltage': X_end_voltage,
            'step': X_step,
            'measured_inputs': measured_inputs,
            'initial_state': initial_state,
            'voltage_unit': voltage_unit,
            'current_unit': current_unit,
            'comments': comments,
            'ax2': None,
            'is_2d_sweep': self.is_2d_sweep,
        }
        initial_state_basic = initial_state.copy()
        
        self.X_start_voltage = self._convert_units(X_start_voltage)
        self.X_end_voltage = self._convert_units(X_end_voltage)
        self.X_step = self._convert_units(X_step)
        self.Y_start_voltage = self._convert_units(Y_start_voltage)
        self.Y_end_voltage = self._convert_units(Y_end_voltage)
        self.Y_step = self._convert_units(Y_step)
        
        self.x_label = self._set_gates_group_label(X_swept_outputs)
        self.y_label = self._set_gates_group_label(Y_swept_outputs)
        self.z_label = self._set_gates_group_label(measured_inputs)
        
        self.comments = comments
        self._set_filename('2D')
        
        with open(f"data/{self.filename}.txt", 'a') as file:
            header = (f"{self.y_label} [{self.voltage_unit}]".rjust(16) +
                      f"{self.x_label} [{self.voltage_unit}]".rjust(16) +
                      f"{self.z_label} [{self.current_unit}]".rjust(16))
            file.write(header + "\n")
            
        # Set up 2D plotting with two subplots
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(20, 6))
        self.ax1.set_xlabel(f"{self.x_label} [{self.voltage_unit}]", fontsize=12)
        self.ax1.set_ylabel(f"{self.y_label} [{self.voltage_unit}]", fontsize=12)
        self.ax2.set_xlabel(f"{self.x_label} [{self.voltage_unit}]", fontsize=12)
        self.ax2.set_ylabel(f"{self.z_label} [{self.current_unit}]", fontsize=12)
        
        X_num = int(round(abs(self.X_end_voltage - self.X_start_voltage) / self.X_step)) + 1
        Y_num = int(round(abs(self.Y_end_voltage - self.Y_start_voltage) / self.Y_step)) + 1
        data = np.full((Y_num, X_num), np.nan)
        
        self._log_params(sweep_type='voltage', status='start')
        
        # Define custom colormap
        colorsbar = ['#02507d', '#ede8e5', '#b5283b']
        cm = LinearSegmentedColormap.from_list('', colorsbar, N=500)

        self.img = self.ax1.imshow(
            data, cmap=cm, aspect='auto', origin='lower',
            extent=[self.X_start_voltage, self.X_end_voltage, self.Y_start_voltage, self.Y_end_voltage], 
            interpolation='none'
            )
        self.fig.patch.set_facecolor('white')

        cbar = self.fig.colorbar(self.img, ax=self.ax1, pad=0.005, extend='both')
        cbar.ax.set_title(rf'         {self.z_label} [{self.current_unit}]', pad=10)  # Colorbar title
        cbar.ax.tick_params(direction='in', width=2, length=5, labelsize=12)  # Colorbar ticks

        self.Y_voltage = self.Y_start_voltage
        idx = 0
        while True:
            # Update the initial state with the current Y voltage for Y-swept outputs
            initial_state = initial_state_basic.copy()
            for Y_gate in Y_swept_outputs.gates:
                initial_state.append([Y_gate, self.Y_voltage, 'V'])
            params['initial_state'] = initial_state
            params['ax2'] = self.ax2
            _, Z_values = self.sweep1D(**params)
            
            data[idx] = Z_values
            self.img.set_data(data)
            
            clim_min = np.nanmin(data[np.isfinite(data)])
            clim_max = np.nanmax(data[np.isfinite(data)])
            self.img.set_clim(clim_min, clim_max)
            barticks = np.linspace(clim_min, clim_max, 5)
            cbar.set_ticks(barticks) 
            cbar.ax.set_yticklabels([f"{t:.2f}" for t in barticks]) 
            cbar.update_normal(self.img)
            self.fig.canvas.draw_idle()
            
            idx += 1
            if (self.Y_start_voltage < self.Y_end_voltage and self.Y_voltage > self.Y_end_voltage - 1e-6) or (
                    self.Y_start_voltage > self.Y_end_voltage and self.Y_voltage < self.Y_end_voltage + 1e-6):
                break
            self.Y_voltage = self.Y_start_voltage + idx * self.Y_step if self.Y_start_voltage < self.Y_end_voltage else self.Y_start_voltage - idx * self.Y_step
            
        plt.ioff()
        print("[INFO] Data collection complete. ")
        plt.close('all')
        self._log_params(sweep_type='voltage', status='end')
        
        # Generate final 2D plot and save the figure
        viz = Visualizer()
        viz.viz2D(f"{self.filename}.txt", is_show=is_show)
        
    def sweepTime(self, 
                  measured_inputs: GatesGroup, 
                  total_time: float,
                  time_step: float, 
                  initial_state: list, 
                  comments: str = None) -> None:
        """
        Perform a time-based sweep by recording current measurements over a specified duration.

        Args:
            measured_inputs (GatesGroup): Group of input gates for measurement.
            total_time (float): Total duration of the sweep in seconds.
            time_step (float): Time interval between measurements in seconds.
            initial_state (list): List of tuples (gate, init_voltage) for initial state.
            comments (str): Additional comments for logging.
        """
        self.x_label = 'time'
        self.z_label = self._set_gates_group_label(measured_inputs)
        self.comments = comments
        self._set_filename('time')

        self.total_time = total_time
        self.time_step = time_step

        pbar = tqdm(total=len(self.outputs.gates), desc="[INFO] Ramping voltage", ncols=80,
                    leave=True)
        
        # Ramp outputs: turn off gates not in the initial state
        idle_gates = [gate for gate in self.outputs.gates if gate not in [state[0] for state in initial_state]]
        GatesGroup(idle_gates).turn_off()
        pbar.update(len(idle_gates))

        # Set initial state for designated gates
        for gate, init_voltage in initial_state:
            gate.voltage(init_voltage, False)

        # Wait for initial voltages to stabilize
        while not all([gate.is_at_target_voltage(voltage) for gate, voltage in initial_state]):
            time.sleep(0.1)
        pbar.update(len(initial_state))
        pbar.close()
        time.sleep(0.1)

        # Set up plotting for time sweep
        fig, ax = plt.subplots(figsize=(12, 7))
        lines, = ax.plot([], [])
        ax.set_xlabel(f"{self.x_label} [s]")
        ax.set_ylabel(f"{self.z_label} [uA]")

        self.times = []
        self.currents = []

        # Log time sweep parameters
        self._log_params(sweep_type='time', status='start')
        with open(f"data/{self.filename}.txt", 'a') as file:
            header = f"{'time [s]':>24} {self.z_label + ' [uA]':>24}"
            file.write(header + "\n")

        total_steps = int(self.total_time // self.time_step)
        pbar = tqdm(total=total_steps, desc="[INFO] Sweeping", ncols=80, leave=True)  # progress bar
        frame = 0
        initial_time = time.time()
        time_list = []
        
        print("[INFO] Start recording time sweep.")
        while True:
            current_elapsed = time.time() - initial_time
            time_list.append(current_elapsed)
            current = measured_inputs.gates[0].read_current()
            self.currents.append(current)
            
            ax.set_xlim(0.0, current_elapsed + self.time_step)
            curr_min, curr_max = min(self.currents), max(self.currents)
            if curr_min == curr_max:
                curr_min -= 0.001
                curr_max += 0.001
            ax.set_ylim(curr_min - (curr_max - curr_min) / 4,
                        curr_max + (curr_max - curr_min) / 4)
            lines.set_data(time_list, self.currents)

            plt.draw()
            plt.pause(0.1)
            frame += 1
            pbar.update(1)

            with open(f"data/{self.filename}.txt", 'a') as file:
                file.write(f"{current_elapsed:>24.2f} {current:>24.16f} \n")
            
            # Wait until the next time step
            while time.time() - initial_time < time_list[-1] + time_step:
                time.sleep(time_step / 100)
            
            if current_elapsed >= total_time:
                pbar.close()
                break

        plt.savefig(f"figures/{self.filename}.png", dpi=300)
        print("[INFO] Data collection complete and figure saved. \n")
        self._log_params(sweep_type='time', status='end')
