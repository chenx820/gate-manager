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
                 amplification: float = None, 
                 temperature: str = None, 
                 device: str = None) -> None:
        self.outputs = outputs
        self.inputs = inputs
        self.amplification = amplification
        self.temperature = temperature
        self.device = device

        # Labels and file metadata
        self.x_label = None
        self.y_label = None
        self.z_label = None
        self.comments = None
        self.filename = None

        # Sweep configuration
        self.start_voltage = None
        self.end_voltage = None
        self.step = None
        self.total_time = None
        self.time_step = None

        # Measurement data
        self.voltage = None
        self.voltages = []
        self.currents = []
        self.is_2d_sweep = False
        
        # Units
        self.voltage_unit = None
        self.current_unit = None
        self.voltage_scale = 1
        self.current_scale = 1
        
    def _set_units(self, voltage_unit: str = 'V', current_unit: str = 'uA') -> None:
        """Set voltage and current units."""
        unit_map = {'V': 1, 'mV': 1e3, 'uV': 1e6}
        self.voltage_scale = unit_map.get(voltage_unit, 1)
        
        unit_map = {'mA': 1e-3, 'uA': 1, 'nA': 1e3, 'pA': 1e6}
        self.current_scale = unit_map.get(current_unit, 1)


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
        filepath = os.path.join(os.getcwd(), base_filename)
        if not os.path.isfile(filepath + '.txt'):
            return base_filename
        
        counter = 2
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
                    f"---- Run started at {self.start_time.strftime('%Y-%m-%d %H:%M:%S')} ----\n")
                file.write(f"{'Filename: ':>16} {self.filename}.txt \n")
                file.write(f"{'Device: ':>16} {self.device} \n")
                file.write(f"{'Voltage Unit: ':>16} {self.voltage_unit} \n")
                file.write(f"{'Current Unit: ':>16} {self.current_unit} \n")
                file.write(f"{'Measured Input: ':>16} {self.z_label} \n")
                file.write("\n")
                file.write(f"{'X Swept Gates: ':>16} {self.x_label} \n")
                if sweep_type == 'voltage':
                    file.write(f"{'Start Voltage: ':>16} {self.start_voltage * self.voltage_scale:>24.16f} [{self.voltage_unit}] \n")
                    file.write(f"{'End Voltage: ':>16} {self.end_voltage * self.voltage_scale:>24.16f} [{self.voltage_unit}] \n")
                    file.write(f"{'Step Size: ':>16} {self.step * self.voltage_scale:24.16f} [{self.voltage_unit}] \n")
                    file.write("\n")
                if self.is_2d_sweep:
                    file.write(f"{'Y Swept Gates: ':>16} {self.y_label} \n")
                    file.write(f"{'Start Voltage: ':>16} {self.Y_voltage * self.voltage_scale:>24.16f} [{self.voltage_unit}] \n")
                    file.write(f"{'End Voltage: ':>16} {self.Y_end_voltage * self.voltage_scale:>24.16f} [{self.voltage_unit}] \n")
                    file.write(f"{'Step Size: ':>16} {self.Y_step * self.voltage_scale:24.16f} [{self.voltage_unit}] \n")
                    file.write("\n")
                if sweep_type == 'time':
                    file.write(f"{'Total Time: ':>16} {self.total_time:>24.2f} [s] \n")
                    file.write(f"{'Time Step: ':>16} {self.time_step:>24.2f} [s] \n")
                    file.write("\n")
                file.write("Initial Voltages of all outputs before sweep: \n")
                for output_gate in self.outputs.gates:
                    file.write(
                        f"{' & '.join(line.label for line in output_gate.lines):>60} {output_gate.voltage() * self.voltage_scale:>24.16f} [{self.voltage_unit}] {output_gate.source.label:>16} \n")
                file.write("\n")
        elif status == 'end':
            total_time_elapsed = datetime.now() - self.start_time
            hours, remainder = divmod(total_time_elapsed.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            with open(f"{self.log_filename}.txt", 'a') as file:
                file.write(f"Total Time: {int(hours)}h {int(minutes)}m {int(seconds)}s \n")
                file.write(
                    f"---- Run ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ----\n")

    def sweep1D(self, 
                swept_outputs: GatesGroup, 
                measured_inputs: GatesGroup, 
                start_voltage: float, 
                end_voltage: float,
                step: float, 
                initial_state: list = None, 
                voltage_unit: str='V',
                current_unit: str='uA',
                comments: str = None, 
                ax2=None, 
                is_2d_sweep: bool = False):
        """
        Perform a 1D voltage sweep and generate an animated plot.

        Args:
            swept_outputs (GatesGroup): Group of output gates to sweep.
            measured_inputs (GatesGroup): Group of input gates for current measurement.
            start_voltage (float): Starting voltage.
            end_voltage (float): Ending voltage.
            step (float): Voltage increment for each step.
            initial_state (list): List of tuples (gate, init_voltage) for initial state.
            voltage_unit (str): Unit for voltage.
            current_unit (str): Unit for current.
            comments (str): Additional comments for logging.
            ax2: Optional axis for plotting if already provided.
            is_2d_sweep (bool): Flag indicating whether this sweep is part of a 2D sweep.

        Returns:
            tuple: (voltages, current_values) if is_2d_sweep is True, else None.
        """
        if step < 0:
            raise ValueError("Step size must be positive.")
        
        # Set sweep labels and units
        self.x_label = self._set_gates_group_label(swept_outputs)
        self.z_label = self._set_gates_group_label(measured_inputs)
        self.voltage_unit = voltage_unit
        self.current_unit = current_unit
        self.comments = comments
        self.ax2 = ax2
        self.is_2d_sweep = is_2d_sweep
        
        self._set_units(self.voltage_unit, self.current_unit)
        if not self.is_2d_sweep:
            self._set_filename('1D')

        self.start_voltage = start_voltage
        self.end_voltage = end_voltage
        self.step = step

        pbar = tqdm(total=len(self.outputs.gates) + len(swept_outputs.gates), desc="[INFO] Ramping voltage", ncols=80,
                    leave=True)
        
        # Ramp outputs: turn off idle gates first
        idle_gates = [gate for gate in self.outputs.gates if gate not in [state[0] for state in initial_state]]
        GatesGroup(idle_gates).turn_off()
        pbar.update(len(idle_gates))

        # Set initial state for designated gates
        for gate, init_volt in initial_state:
            gate.voltage(init_volt, is_wait=False)

        # Wait until all initial voltages stabilize
        while not all([gate.is_at_target_voltage(voltage) for gate, voltage in initial_state]):
            time.sleep(0.1)
        pbar.update(len(initial_state))

        # Set swept outputs to the starting voltage
        swept_outputs.voltage(start_voltage)
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
            self.ax2.set_title(f"{self.y_label} Voltage: {self.Y_voltage * self.voltage_scale:.3f} [{self.voltage_unit}]")
        self.ax2.set_xlabel(f"{self.x_label} [{self.voltage_unit}]")
        self.ax2.set_ylabel(f"{self.z_label} [{self.current_unit}]")
        
        self.voltages = []
        self.currents = []
        self.voltage = self.start_voltage

        # Log sweep parameters (for non-2D sweeps)
        if not self.is_2d_sweep:
            self._log_params(status='start')
            with open(f"{self.filename}.txt", 'a') as file:
                header = (f"{self.x_label} [{self.voltage_unit}]".rjust(24) + 
                          f"{self.z_label} [{self.current_unit}]".rjust(24))
                file.write(header + "\n")

        print(
            f"[INFO] Start sweeping {self.x_label} from {self.start_voltage*self.voltage_scale} " 
            f"[{self.voltage_unit}] to {self.end_voltage*self.voltage_scale} [{self.voltage_unit}].")
        

        self.lines, = self.ax2.plot([], [])
        total_steps = round(abs(self.end_voltage - self.start_voltage) / self.step + 1)
        pbar = tqdm(total=total_steps, desc="[INFO] Sweeping", ncols=80, leave=True) 
        frame = 0
        
        while True:
            swept_outputs.voltage(self.voltage)
            self.voltages.append(self.voltage * self.voltage_scale)
            
            # Read current from the first measured input (extend as needed)
            current = measured_inputs.gates[0].read_current(self.amplification) * self.current_scale
            self.currents.append(current)
            
            # Update plot limits and data
            self.ax2.set_xlim(
                min(self.voltages) - self.step*self.voltage_scale, 
                max(self.voltages) + self.step*self.voltage_scale
                )
            curr_min = min(self.currents)
            curr_max = max(self.currents)
            if curr_min == curr_max:
                curr_min -= 0.001
                curr_max += 0.001
            self.ax2.set_ylim(min(self.currents) - (curr_max - curr_min) / 4,
                              max(self.currents) + (curr_max - curr_min) / 4)
            self.lines.set_data(self.voltages, self.currents)

            plt.draw()
            plt.pause(0.01)
            frame += 1
            pbar.update(1)

            with open(f"{self.filename}.txt", 'a') as file:
                if self.is_2d_sweep:
                    file.write(f"{self.Y_voltage * self.voltage_scale:>24.8f} " 
                               f"{self.voltage * self.voltage_scale:>24.8f} "
                               f"{current:>24.8f} \n")
                else: 
                    file.write(f"{self.voltage * self.voltage_scale:>24.8f} "
                               f"{current:>24.8f} \n")
                    
            # Check if sweep is complete    
            if (self.start_voltage < self.end_voltage and self.voltage > self.end_voltage - 1e-6) or (
                    self.start_voltage > self.end_voltage and self.voltage < self.end_voltage + 1e-6):
                pbar.close()
                break
            self.voltage = self.start_voltage + frame * self.step \
                if self.start_voltage < self.end_voltage \
                else self.start_voltage - frame * self.step
        
        if self.is_2d_sweep:
            print("\n")
            return self.voltages, self.currents
        else:
            plt.ioff()
            plt.tight_layout()
            plt.savefig(f"{self.filename}.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("[INFO] Data collection complete and figure saved. \n")
            self._log_params(status='end')
            

    def sweep2D(self, 
                X_swept_outputs: GatesGroup, 
                X_start_voltage: float, 
                X_end_voltage: float, 
                X_step: float,
                Y_swept_outputs: GatesGroup, 
                Y_start_voltage: float, 
                Y_end_voltage: float, 
                Y_step: float,
                measured_inputs: GatesGroup, 
                initial_state: list, 
                voltage_unit: str = 'V',
                current_unit: str = 'uA',
                comments: str = None):
        """
        Perform a 2D voltage sweep over two axes by sweeping one set of outputs for each voltage
        setting of another set.

        Args:
            X_swept_outputs (GatesGroup): Gates to sweep along the X axis.
            X_start_voltage (float): Starting voltage for X axis.
            X_end_voltage (float): Ending voltage for X axis.
            X_step (float): Voltage step for X axis.
            Y_swept_outputs (GatesGroup): Gates to sweep along the Y axis.
            Y_start_voltage (float): Starting voltage for Y axis.
            Y_end_voltage (float): Ending voltage for Y axis.
            Y_step (float): Voltage step for Y axis.
            measured_inputs (GatesGroup): Group of input gates for measurements.
            initial_state (list): List of tuples (gate, init_voltage) for initial settings.
            voltage_unit (str): Voltage unit.
            current_unit (str): Current unit.
            comments (str): Additional comments for logging.
        """
        self.X_start_voltage = X_start_voltage
        self.X_end_voltage = X_end_voltage
        self.X_step = X_step
        self.Y_start_voltage = Y_start_voltage
        self.Y_end_voltage = Y_end_voltage
        self.Y_step = Y_step
        
        self.voltage_unit = voltage_unit
        self.current_unit = current_unit
        self.is_2d_sweep = True
        
        self._set_units(self.voltage_unit, self.current_unit)
        
        # Prepare parameters for the 1D sweep call
        params = {
            # here we use the variable name for the gate which is okay
            'swept_outputs': X_swept_outputs,
            'start_voltage': self.X_start_voltage,
            'end_voltage': self.X_end_voltage,
            'step': self.X_step,
            'measured_inputs': measured_inputs,
            'initial_state': initial_state,
            'voltage_unit': voltage_unit,
            'current_unit': current_unit,
            'comments': comments,
            'ax2': None,
            'is_2d_sweep': self.is_2d_sweep
        }
        initial_state_basic = initial_state.copy()
        
        self.x_label = self._set_gates_group_label(X_swept_outputs)
        self.y_label = self._set_gates_group_label(Y_swept_outputs)
        self.z_label = self._set_gates_group_label(measured_inputs)
        
        self.comments = comments
        self._set_filename('2D')
        
        with open(f"{self.filename}.txt", 'a') as file:
            header = (f"{self.y_label} [{self.voltage_unit}]".rjust(24) +
                      f"{self.x_label} [{self.voltage_unit}]".rjust(24) +
                      f"{self.z_label} [{self.current_unit}]".rjust(24))
            file.write(header + "\n")
            
        # Set up 2D plotting with two subplots
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(20, 6))
        self.ax1.set_xlabel(f"{self.x_label} [{self.voltage_unit}]", fontsize=12)
        self.ax1.set_ylabel(f"{self.y_label} [{self.voltage_unit}]", fontsize=12)
        self.ax2.set_xlabel(f"{self.x_label} [{self.voltage_unit}]", fontsize=12)
        self.ax2.set_ylabel(f"{self.z_label} [{self.current_unit}]", fontsize=12)
        
        X_num = int(round(abs(self.X_end_voltage - self.X_start_voltage) / X_step)) + 1
        Y_num = int(round(abs(self.Y_end_voltage - self.Y_start_voltage) / Y_step)) + 1
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

        Y_voltage = Y_start_voltage
        idx = 0
        while True:
            # Update the initial state with the current Y voltage for Y-swept outputs
            initial_state = initial_state_basic.copy()
            self.Y_voltage = Y_voltage
            for Y_gate in Y_swept_outputs.gates:
                initial_state.append([Y_gate, Y_voltage])
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
            if (Y_start_voltage < Y_end_voltage and Y_voltage > Y_end_voltage - 1e-6) or (
                    Y_start_voltage > Y_end_voltage and Y_voltage < Y_end_voltage + 1e-6):
                break
            Y_voltage = Y_start_voltage + idx * Y_step if Y_start_voltage < Y_end_voltage else Y_start_voltage - idx * Y_step
            
        plt.ioff()
        print("[INFO] Data collection complete. ")
        plt.close('all')
        self._log_params(sweep_type='voltage', status='end')
        
        # Generate final 2D plot and save the figure
        viz = Visualizer()
        viz.viz2D(f"{self.filename}.txt")
        
    def sweepTime(self, 
                  measured_inputs: GatesGroup, 
                  total_time: float,
                  time_step: float, 
                  initial_state: list = None, 
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
        with open(f"{self.filename}.txt", 'a') as file:
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
            current = measured_inputs.gates[0].read_current(self.amplification)
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

            with open(f"{self.filename}.txt", 'a') as file:
                file.write(f"{current_elapsed:>24.2f} {current:>24.16f} \n")
            
            # Wait until the next time step
            while time.time() - initial_time < time_list[-1] + time_step:
                time.sleep(time_step / 100)
            
            if current_elapsed >= total_time:
                pbar.close()
                break

        plt.savefig(f"{self.filename}.png", dpi=300)
        print("[INFO] Data collection complete and figure saved. \n")
        self._log_params(sweep_type='time', status='end')
