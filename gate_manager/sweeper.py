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

from decimal import Decimal
from datetime import datetime
import time
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from .gate import Gate, GatesGroup


class Sweeper:
    """
    Sweeper class to perform and log voltage sweeps on defined gates.

    Attributes:
        outputs (GatesGroup): Group of output gates.
        inputs (GatesGroup): Group of input gates used for measurements.
        slew_rate (float): Rate of change of voltage [V/s].
        amplification (float): Amplification factor for current measurements.
        temperature (str): Temperature during the sweep.
        device (str): Identifier for the device under test.
        x_label (str): Label for x-axis in plots.
        y_label (str): Label for y-axis in plots.
        comments (str): Additional comments for the experiment.
        filename (str): Base filename for saving results.
        filename_2d (str): Base filename for saving 2D sweep results.
        start_voltage (Decimal): Starting voltage for the sweep.
        end_voltage (Decimal): Ending voltage for the sweep.
        step (Decimal): Voltage increment per step.
        total_time (float): Total duration for a time sweep.
        time_step (float): Time interval between measurements.
        voltage (Decimal): Current voltage during the sweep.
        voltages (list): Recorded voltage values.
        currents (list): Recorded current measurements.
        is_save_file (bool): Flag to enable file saving.
    """

    def __init__(self, outputs=None, inputs=None, slew_rate=None, amplification=None, temperature=None, device=None):
        self.outputs = outputs
        self.inputs = inputs
        self.slew_rate = slew_rate
        self.amplification = amplification
        self.temperature = temperature
        self.device = device

        # Labels and file metadata
        self.x_label = None
        self.y_label = None
        self.comments = None
        self.filename = None
        self.filename_2d = None

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
        self.is_save_file = False

    def set_gates_group_label(self, gates_group):
        """
        Generate a label by combining the labels from all lines in a group of gates.

        Args:
            gates_group (GatesGroup): The group of gates.

        Returns:
            str: Combined label from all gate lines.
        """
        return " & ".join(line.label for gate in gates_group.gates for line in gate.lines)

    def set_gate_label(self, gate):
        """
        Generate a label for a single gate by combining its line labels.

        Args:
            gate (Gate): The gate to label.

        Returns:
            str: Combined label from the gate's lines.
        """
        return " & ".join(line.label for line in gate.lines)

    def set_filename(self):
        """
        Generate a unique filename based on temperature, axis labels, and comments.
        This is used for saving measurement data.
        """
        if self.is_save_file:
            current_dir = os.getcwd()
            base_filename = f"{self.temperature}_[{self.y_label}]_vs_[{self.x_label}]"
            if self.comments:
                base_filename += f"_{self.comments}"
            filepath = os.path.join(current_dir, base_filename)
            # If a file already exists, add a run counter
            if os.path.isfile(filepath + '.txt'):
                counter = 2
                while os.path.isfile(f"{filepath}_run{counter}.txt"):
                    counter += 1
                base_filename = f"{base_filename}_run{counter}"
            self.filename = base_filename
        elif self.filename_2d is not None:
            self.filename = self.filename_2d
            

    def log_params(self, type='voltage') -> None:
        """
        Log sweep parameters and experimental metadata to a file.

        Args:
            sweep_type (str): Type of sweep ('voltage' or 'time') to log specific parameters.
        """
        log_filename = "log"
        if self.comments:
            log_filename += f"_{self.comments}"
        with open(f"{log_filename}.txt", 'a') as file:
            file.write(
                f"--- Run started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            file.write(f"{'Filename: ':>16} {self.filename}.txt \n")
            file.write(f"{'Device: ':>16} {self.device} \n")
            file.write(f"{'Amplifier: ':>16} {self.amplification} \n")
            file.write(f"{'Slew Rate: ':>16} {self.slew_rate} [V/s] \n")
            file.write(f"{'Swept Gates: ':>16} {self.x_label} \n")
            file.write(f"{'Measured Input: ':>16} {self.y_label} \n")
            if type == 'voltage':
                file.write(f"{'Start Voltage: ':>16} {self.start_voltage:>24.16f} [V] \n")
                file.write(f"{'End Voltage: ':>16} {self.end_voltage:>24.16f} [V] \n")
                file.write(f"{'Step Size: ':>16} {self.step:24.16f} [V] \n")
            if type == 'time':
                file.write(f"{'Total time: ':>16} {self.total_time:>24.2f} [s] \n")
                file.write(f"{'Time Step: ':>16} {self.time_step:>24.2f} [s] \n")
            file.write("Initial Voltages of all outputs before sweep: \n")
            for output_gate in self.outputs.gates:
                file.write(
                    f"{output_gate.source.label:>16} {output_gate.voltage():>24.16f} [V] {' & '.join(line.label for line in output_gate.lines):>80} \n")
            file.write("\n")

    def sweep1D(self, swept_outputs: GatesGroup, measured_inputs: GatesGroup, start_voltage: float, end_voltage: float,
                step: float, initial_state: list = None, comments: str = None, is_save_file: bool = True, ax2=None, is_2d_sweep=False) -> None:
        """
        Perform a 1D voltage sweep and generate an animated plot.

        Args:
            swept_outputs (GatesGroup): Group of output gates to sweep.
            measured_inputs (GatesGroup): Group of input gates for current measurement.
            start_voltage (float): Starting voltage.
            end_voltage (float): Ending voltage.
            step (float): Voltage increment for each step.
            initial_state (list): List of tuples (gate, init_voltage) for initial state.
            comments (str): Additional comments for logging.
            is_save_file (bool): Flag to save results to a file.
            ax2: Optional axis for plotting if already provided.
            is_2d_sweep (bool): Flag indicating whether this sweep is part of a 2D sweep.

        Returns:
            tuple: (voltages, current_values) if is_2d_sweep is True, else None.
        """
        if step < 0:
            raise ValueError("Step size must be positive.")
        # Set sweep labels
        self.x_label = self.set_gates_group_label(swept_outputs)
        self.y_label = self.set_gates_group_label(measured_inputs)
        self.comments = comments
        self.is_save_file = is_save_file
        
        self.set_filename()

        self.start_voltage = Decimal(start_voltage)
        self.end_voltage = Decimal(end_voltage)
        self.step = Decimal(step)

        # Ramp outputs: turn off idle gates first
        pbar = tqdm(total=len(self.outputs.gates) + len(swept_outputs.gates), desc="[INFO] Ramping voltage", ncols=80,
                    leave=True)
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


        # To do: If there is more than one measured input? 
        
        # Set up plotting
        if not is_2d_sweep:
            plt.ion()
            fig, ax2 = plt.subplots(1, 1, figsize=(8, 6))
            fig.suptitle(f"1D Sweep: {self.x_label} vs {self.y_label}")
        else:
            ax2.clear()
            ax2.set_title(f"{self.y_label} Voltage: {self.Y_voltage} V")
            ax2.set_xlabel(f"{self.x_label} [V]")
            ax2.set_ylabel(f"{self.y_label} [uA]")
        self.ax2 = ax2
        
        self.voltages = []
        self.currents = []
        self.voltage = self.start_voltage

        # Log sweep parameters
        self.log_params()

        print(
            f"[INFO] Start sweeping {self.x_label} from {float(self.start_voltage)} [V] to {float(self.end_voltage)} [V].")
        
        if self.is_save_file:
            with open(f"{self.filename}.txt", 'a') as file:
                header = f"{self.x_label} [V]".rjust(24) + f"{self.y_label} [uA]".rjust(24)
                file.write(header + "\n")

        self.lines, = ax2.plot([], [])
        
        # Execute sweep
        total = round(abs(self.end_voltage - self.start_voltage) / self.step + 1)
        pbar = tqdm(total=total, desc="[INFO] Sweeping", ncols=80, leave=True) 
        frame = 0
        
        while True:
            swept_outputs.voltage(self.voltage)
            self.voltages.append(self.voltage)
            
            # Read current from the first measured input (extend as needed)
            current = measured_inputs.gates[0].read_current(self.amplification)
            self.currents.append(current)
            
            # Update plot limits and data
            curr_min = min(self.currents)
            curr_max = max(self.currents)
            if curr_min == curr_max:
                curr_min -= Decimal(0.001)
                curr_max += Decimal(0.001)
                self.ax2.set_ylim(curr_min, curr_max)
            else:
                self.ax2.set_ylim(min(self.currents) - (max(self.currents)-min(self.currents))/3,
                              max(self.currents) + (max(self.currents)-min(self.currents))/3)
            self.ax2.set_xlim(min(self.voltages) - self.step, max(self.voltages) + self.step)
            
            self.lines.set_data(self.voltages, self.currents)

            plt.draw()
            plt.pause(0.01)
            frame += 1
            pbar.update(1)

            if self.is_save_file:
                with open(f"{self.filename}.txt", 'a') as file:
                    file.write(f"{self.voltage:>24.8f}{current:>24.8f} \n")
            if is_2d_sweep:
               with open(f"{self.filename_2d}.txt", 'a') as file:
                   file.write(f"{self.Y_voltage:>24.8f}{self.voltage:>24.8f}{current:>24.16f} \n")
            
            # Check if sweep is complete    
            if (self.start_voltage < self.end_voltage and self.voltage > self.end_voltage - Decimal(1e-6)) or (
                    self.start_voltage > self.end_voltage and self.voltage < self.end_voltage + Decimal(1e-6)):
                pbar.close()
                break
            self.voltage = self.start_voltage + frame * self.step \
                if self.start_voltage < self.end_voltage \
                else self.start_voltage - frame * self.step
        
        if not is_2d_sweep:
            plt.savefig(f"{self.filename}.png", dpi=300)
            plt.close()
            print("[INFO] Data collection complete and figure saved. \n")
        else:
            print("\n")
            return self.voltages, self.currents

    def sweep2D(self, X_swept_outputs: GatesGroup, X_start_voltage: float, X_end_voltage: float, X_step: float,
                Y_swept_outputs: GatesGroup, Y_start_voltage: float, Y_end_voltage: float, Y_step: float,
                measured_inputs: GatesGroup, initial_state: list, comments: str = None):
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
            comments (str): Additional comments for logging.
        """
        params = {
            # here we use the variable name for the gate which is okay
            'swept_outputs': X_swept_outputs,
            'start_voltage': X_start_voltage,
            'end_voltage': X_end_voltage,
            'step': X_step,
            'measured_inputs': measured_inputs,
            'initial_state': initial_state,
            'comments': comments,
            'is_save_file': False
        }
        initial_state_basic = initial_state.copy()
        
        self.x_label_2d = self.set_gates_group_label(X_swept_outputs)
        self.y_label_2d = self.set_gates_group_label(Y_swept_outputs)
        self.z_label_2d = self.set_gates_group_label(measured_inputs)
        
        current_dir = os.getcwd()
        # Set up a unique filename for 2D logging for each measured input
        base_filename = f"{self.temperature}_[{self.z_label_2d}]_vs_[{self.x_label_2d}]_[{self.y_label_2d}]"
        if comments:
            base_filename += f"_{comments}"
        filepath = os.path.join(current_dir, base_filename)
        if os.path.isfile(filepath + '.txt'):
            counter = 2
            while os.path.isfile(f"{filepath}_run{counter}.txt"):
                counter += 1
            base_filename = f"{base_filename}_run{counter}"
        self.filename_2d = base_filename
        with open(f"{self.filename_2d}.txt", 'a') as file:
            header = f"{self.y_label_2d} [V]".rjust(24)
            header += f"{self.x_label_2d} [V]".rjust(24)
            header += f"{self.z_label_2d} [uA]".rjust(24)
            file.write(header + "\n")
            
        # Set up 2D plotting with two subplots
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(16, 6))
        self.fig.suptitle(f"2D Sweep: [{self.z_label_2d}] vs [{self.x_label_2d} & {self.y_label_2d}]")
        self.ax1.set_xlabel(f"{self.x_label_2d} [V]")
        self.ax1.set_ylabel(f"{self.y_label_2d} [V]")
        self.ax2.set_xlabel(f"{self.x_label_2d} [V]")
        self.ax2.set_ylabel(f"{self.z_label_2d} [uA]")
        
        X_num = int(round(abs(X_end_voltage - X_start_voltage) / X_step)) + 1
        Y_num = int(round(abs(Y_end_voltage - Y_start_voltage) / Y_step)) + 1
        data = np.zeros((Y_num, X_num))
        self.im = self.ax1.imshow(data, cmap='bwr', aspect='auto', origin='lower',
                                extent=[X_start_voltage, X_end_voltage, Y_start_voltage, Y_end_voltage])
        plt.colorbar(self.im, ax=self.ax1, label=r"I [uA]")
        data_matrix = []
        
        Y_voltage = Y_start_voltage
        loop = 0
        while True:
            # Update the initial state with the current Y voltage for Y-swept outputs
            initial_state = initial_state_basic.copy()
            self.Y_voltage = Y_voltage
            for Y_gate in Y_swept_outputs.gates:
                initial_state.append([Y_gate, Y_voltage])
            params['initial_state'] = initial_state
            params['ax2'] = self.ax2  
            params['is_2d_sweep'] = True   
            X_values, Z_values = self.sweep1D(**params)
            
            data[loop] = Z_values
            data_matrix.append(Z_values)
            self.im.set_data(data)
            self.im.set_clim(vmin=np.nanmin(data), vmax=np.nanmax(data))
            self.fig.canvas.draw_idle()
            
            loop += 1
            if (Y_start_voltage < Y_end_voltage and Y_voltage > Y_end_voltage - 1e-6) or (
                    Y_start_voltage > Y_end_voltage and Y_voltage < Y_end_voltage + 1e-6):
                break
            Y_voltage = Y_start_voltage + loop * Y_step if Y_start_voltage < Y_end_voltage else Y_start_voltage - loop * Y_step
            
        plt.ioff()
        plt.close()
        print("[INFO] Data collection complete and figure saved. \n")
        
        # Generate final 2D plot and save the figure
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(data, aspect='auto', cmap='bwr', origin='lower',
                       extent=[min(X_start_voltage, X_end_voltage), max(X_start_voltage, X_end_voltage),
                               min(Y_start_voltage, Y_end_voltage), max(Y_start_voltage, Y_end_voltage)])
        plt.colorbar(im, ax=ax, label=r"I [uA]")
        ax.set_title(f"2D Sweep: [{self.z_label_2d}] vs [{self.x_label_2d} & {self.y_label_2d}]")
        ax.set_xlabel(self.set_gates_group_label(X_swept_outputs)+'[V]')
        ax.set_ylabel(self.set_gates_group_label(Y_swept_outputs)+'[V]')
        plt.savefig(f"{self.filename_2d}.png", dpi=500)
        


    def sweepTime(self, measured_inputs: GatesGroup, total_time: float,
                time_step: float, initial_state: list = None, comments: str = None, is_save_file=True) -> None:
        """
        Perform a time-based sweep by recording current measurements over a specified duration.

        Args:
            measured_inputs (GatesGroup): Group of input gates for measurement.
            total_time (float): Total duration of the sweep in seconds.
            time_step (float): Time interval between measurements in seconds.
            initial_state (list): List of tuples (gate, init_voltage) for initial state.
            comments (str): Additional comments for logging.
            is_save_file (bool): Flag to save results to a file.
        """
        self.x_label = 'time'
        self.y_label = self.set_gates_group_label(measured_inputs)
        self.comments = comments
        self.is_save_file = is_save_file
        self.set_filename()

        self.total_time = total_time
        self.time_step = time_step

        # Ramp outputs: turn off gates not in the initial state
        pbar = tqdm(total=len(self.outputs.gates), desc="[INFO] Ramping voltage", ncols=80,
                    leave=True)
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
        fig, ax = plt.subplots(figsize=(8, 6))
        lines, = ax.plot([], [])
        ax.set_xlabel(f"{self.x_label} [s]")
        ax.set_ylabel(f"{self.y_label} [uA]")

        self.times = []
        self.currents = []

        # Log time sweep parameters
        self.log_params("time")

        print("[INFO] Start recording time sweep.")
        with open(f"{self.filename}.txt", 'a') as file:
            header = f"{self.x_label + ' [s]':>24}{self.y_label + ' [uA]':>24}"
            file.write(header + "\n")

        total = self.total_time // self.time_step
        pbar = tqdm(total=total, desc="[INFO] Sweeping", ncols=80, leave=True)  # progress bar
        frame = 0
        initial_time = time.time()
        current_time_list = []
        
        while True:
            self.current_time = time.time() - initial_time
            current_time_list.append(self.current_time)
            current = measured_inputs.gates[0].read_current(self.amplification)
            self.currents.append(current)
            
            curr_min = min(self.currents)
            curr_max = max(self.currents)
            if curr_min == curr_max:
                curr_min -= Decimal(0.001)
                curr_max += Decimal(0.001)
                ax.set_ylim(curr_min, curr_max)
            else:
                ax.set_ylim(min(self.currents) - (max(self.currents)-min(self.currents))/3,
                              max(self.currents) + (max(self.currents)-min(self.currents))/3)
            ax.set_xlim(0.0, self.current_time + self.time_step)
            lines.set_data(current_time_list, self.currents)

            plt.draw()
            plt.pause(0.1)
            frame += 1
            pbar.update(1)

            with open(f"{self.filename}.txt", 'a') as file:
                file.write(f"{self.current_time:>24.2f}{current:>24.16f} \n")
            
            # Wait until the next time step
            while time.time() - initial_time < current_time_list[-1] + time_step:
                time.sleep(time_step / 100)
            
            if self.current_time >= total_time:
                pbar.close()
                break

        plt.savefig(f"{self.filename}.png", dpi=300)
        print("[INFO] Data collection complete and figure saved. \n")
