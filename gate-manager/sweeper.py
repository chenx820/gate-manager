# -*- coding: utf-8 -*-
"""
Sweeper Class for Conducting Voltage Sweeps with Nanonis System.

This module provides the `Sweeper` class to perform 1D and 2D voltage sweeps across a set of gates
using the Nanonis system, logging measurement data and generating animated plots. It enables precise
control of sweep parameters and data recording for experimental analysis.

Classes:
    Sweeper: Conducts voltage sweeps on specified gates, logs results, and generates plots for analysis.

Created on Wed Nov 06 10:46:06 2024
@author: Chen Huang <chen.huang23@imperial.ac.uk>
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
    A class to perform and log voltage sweeps on defined gates.

    Attributes:
        outputs (GatesGroup): Group of gates that serve as outputs.
        inputs (GatesGroup): Group of gates that serve as inputs.
        slew_rate (float): Rate of change of voltage over time [V/s].
        amplification (float): Amplification factor for current measurements.
        temperature (str): Temperature at which the sweep is conducted.
        device (str): Identifier for the device under test.
        x_label (str): Label for x-axis in plots.
        y_label (str): Label for y-axis in plots.
        comments (str): Comments to annotate the experiment.
        filename (str): Filename to save results.
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
        self.y_labels = []
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

    def set_gates_group_label(self, gates_group):
        """Set label by combining labels from all lines in a group of gates."""
        return " & ".join(line.label for gate in gates_group.gates for line in gate.lines)

    def set_gate_label(self, gate):
        """Set label using labels of lines in a single gate."""
        return " & ".join(line.label for line in gate.lines)

    def set_filename(self):
        """Generate a unique filename based on temperature, x/y labels, and comments."""
        if self.is_file:
            current_dir = os.getcwd()
            self.filename = f"{self.temperature}_[{self.y_label}]_vs_[{self.x_label}]"
            if self.comments is not None:
                self.filename = self.filename + '_' + self.comments
            filepath = os.path.join(current_dir, self.filename)
            if os.path.isfile(filepath + '.txt'):
                counter = 2
                while os.path.isfile(f"{filepath}_run{counter}.txt"):
                    counter += 1
                self.filename = f"{self.filename}_run{counter}"
        else:
            self.filename = None
            

    def log_params(self, type='voltage') -> None:
        """Log the parameters of the sweep to a file, capturing experimental metadata."""
        log_filename = "log"
        if self.comments is not None:
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
                step: float, initial_state: list = None, comments: str = None, is_file: bool = True, ax2=None, is_2d=False) -> None:
        """
        Perform a 1D voltage sweep and create an animated plot of the measurement.

        Args:L
            swept_outputs (GatesGroup): Group of output gates to sweep.
            measured_inputs (GatesGroup): Group of input gate for measuring current.
            start_voltage (float): Starting voltage for sweep.
            end_voltage (float): End voltage for sweep.
            step (float): Increment for each voltage step.
            initial_state (list): List of initial voltages for gates.
            comments (str): Additional comments for logging.
            is_file (bool): Write 1d data file or not.
        """
        self.x_label = self.set_gates_group_label(swept_outputs)
        self.y_label = self.set_gates_group_label(measured_inputs)
        self.y_labels = [self.set_gate_label(measured_input) for measured_input in measured_inputs.gates]
        self.comments = comments
        self.is_file = is_file
        
        self.set_filename()

        self.start_voltage = Decimal(start_voltage)
        self.end_voltage = Decimal(end_voltage)
        self.step = Decimal(step)

        # Progress bar for ramping up
        pbar = tqdm(total=len(self.outputs.gates) + len(swept_outputs.gates), desc="[INFO] Ramping voltage", ncols=80,
                    leave=True)

        idle_gates = []
        for gate in self.outputs.gates:
            if gate not in [state[0] for state in initial_state]:
                idle_gates.append(gate)
        GatesGroup(idle_gates).turn_off()
        pbar.update(len(idle_gates))

        # Set initial voltage for each gate in initial_state
        for gate, initial_voltage in initial_state:
            gate.voltage(initial_voltage, False)

        # Wait for initial voltages to stabilize
        while not all([gate.is_at_target_voltage(voltage) for gate, voltage in initial_state]):
            time.sleep(0.1)
        pbar.update(len(initial_state))

        # Initialize sweep and plot
        swept_outputs.voltage(start_voltage)
        pbar.update(len(swept_outputs.gates))
        pbar.close()
        time.sleep(1)

        # Set up plot
        num = len(measured_inputs.gates)
        #fig, axes = plt.subplots(num, 1, figsize=(8, 5*num))
        
        
        
        # Initialize real-time plotting
        if not is_2d:
            # 独立运行sweep1D时创建新图形
            plt.ion()
            fig, ax2 = plt.subplots(1, 1, figsize=(8, 6))
            fig.suptitle(f"1D Sweep: {self.x_label} vs {self.y_label}")
        else:
            ax2.clear()
            ax2.set_title(f"{self.y_label_2d} Voltage: {self.Y_voltage} V")
            ax2.set_xlabel(f"{self.x_label} [V]")
            ax2.set_ylabel(f"{self.y_label} [uA]")
        self.ax2 = ax2
        
        self.voltages = []
        self.currents = [[] for _ in range(num)]
        self.voltage = self.start_voltage

        # Record parameters
        self.log_params()

        # Start data collection
        print(
            f"[INFO] Start sweeping {self.x_label} from {float(self.start_voltage)} [V] to {float(self.end_voltage)} [V].")
        
        if self.is_file:
            with open(f"{self.filename}.txt", 'a') as file:
                file.write(f"{self.x_label + ' [V]':>24}"
                           + "".join(f"{label + ' [uA]':>24}" for label in self.y_labels)
                           + "\n"
                           )

        self.lines, = ax2.plot([], [])
        
        # Execute sweep and record data
        total = round(abs(self.end_voltage - self.start_voltage) / self.step + 1)
        pbar = tqdm(total=total, desc="[INFO] Sweeping", ncols=80, leave=True)  # progress bar
        frame = 0
        while True:
            swept_outputs.voltage(self.voltage)
            self.voltages.append(self.voltage)
            current = [None for _ in range(num)]

            current[0] = measured_inputs.gates[0].read_current(self.amplification)
            self.currents[0].append(current[0])
            self.ax2.set_xlim(min(self.voltages) - self.step, max(self.voltages) + self.step)
            self.ax2.set_ylim(min(self.currents[0]) - (max(self.currents[0])-min(self.currents[0]))/3,
                              max(self.currents[0]) + (max(self.currents[0])-min(self.currents[0]))/3)
            self.lines.set_data(self.voltages, self.currents[0])

            plt.draw()
            plt.pause(0.01)
            frame += 1
            pbar.update(1)

            if self.filename:
                with open(f"{self.filename}.txt", 'a') as file:
                    file.write(f"{self.voltage:>24.8f}"
                               + "".join(f"{curr:>24.16f}" for curr in current)
                               + " \n"
                               )
            if self.filename_2d:
               with open(f"{self.filename_2d}.txt", 'a') as file:
                   file.write(f"{self.Y_voltage:>24.8f}"
                              + f"{self.voltage:>24.8f}"
                              + "".join(f"{curr:>24.16f}" for curr in current)
                              + " \n"
                              )
            if (self.start_voltage < self.end_voltage and self.voltage > self.end_voltage - Decimal(1e-6)) or (
                    self.start_voltage > self.end_voltage and self.voltage < self.end_voltage + Decimal(1e-6)):
                pbar.close()
                break
            self.voltage = self.start_voltage + frame * self.step \
                if self.start_voltage < self.end_voltage \
                else self.start_voltage - frame * self.step
                
        print("[INFO] Data collection complete and figure saved. \n")
        
        if not is_2d:
            plt.savefig(f"{self.filename}.png", dpi=300)
            print("[INFO] Data collection complete and figure saved. \n")
        else:
            return self.voltages, self.currents[0]

    def sweep2D(self, X_swept_outputs: GatesGroup, X_start_voltage: float, X_end_voltage: float, X_step: float,
                Y_swept_outputs: GatesGroup, Y_start_voltage: float, Y_end_voltage: float, Y_step: float,
                measured_inputs: GatesGroup, initial_state: list, comments: str = None, show_2D: bool=True):
        """
        Perform a 2D voltage sweep over two terminals, sweeping X_swept_terminal for each Y_swept_terminal voltage.

        Args:
            X_swept_outputs (GatesGroup): Gates to sweep over X axis.
            X_start_voltage (float): Start voltage for X axis sweep.
            X_end_voltage (float): End voltage for X axis sweep.
            X_step (float): Step size for X axis sweep.
            Y_swept_outputs (GatesGroup): Gates to sweep over Y axis.
            Y_start_voltage (float): Start voltage for Y axis sweep.
            Y_end_voltage (float): End voltage for Y axis sweep.
            Y_step (float): Step size for Y axis sweep.
            measured_input (GatesGroup): Gate to measure input.
            initial_state (list): Initial voltages for gates.
            comments (str): Additional comments for logging.
        """
        Y_voltage = Y_start_voltage
        loop = 0
        params = {
            # here we use the variable name for the gate which is okay
            'swept_outputs': X_swept_outputs,
            'start_voltage': X_start_voltage,
            'end_voltage': X_end_voltage,
            'step': X_step,
            'measured_inputs': measured_inputs,
            'initial_state': initial_state,
            'comments': comments,
            'is_file': False
        }
        initial_state_basic = initial_state.copy()
        
        self.x_label_2d = self.set_gates_group_label(X_swept_outputs)
        self.y_label_2d = self.set_gates_group_label(Y_swept_outputs)
        self.z_label_2d = self.set_gates_group_label(measured_inputs)
        self.z_labels_2d = [self.set_gate_label(measured_input) for measured_input in measured_inputs.gates]
        
        current_dir = os.getcwd()
        for measured_input in measured_inputs.gates:
            self.filename_2d = f"{self.temperature}_[{self.set_gate_label(measured_input)}]_vs_[{self.x_label_2d}]_[{self.y_label_2d}]"
            if comments is not None:
                self.filename_2d = self.filename_2d + '_' + comments
            filepath = os.path.join(current_dir, self.filename_2d)
            if os.path.isfile(filepath + '.txt'):
                counter = 2
                while os.path.isfile(f"{filepath}_run{counter}.txt"):
                    counter += 1
                self.filename_2d = f"{self.filename_2d}_run{counter}"
                
            with open(f"{self.filename_2d}.txt", 'a') as file:
                file.write(f"{self.set_gates_group_label(Y_swept_outputs) + ' [V]':>24}"
                           + f"{self.set_gates_group_label(X_swept_outputs) + ' [V]':>24}"
                           + "".join(f"{label + ' [uA]':>24}" for label in self.z_labels_2d)
                           + "\n"
                           )
        
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(16, 6))
        self.fig.suptitle(f"2D Sweep: [{self.z_label_2d}] vs [{self.x_label_2d} & {self.y_label_2d}]")
        self.ax1.set_xlabel(f"{self.x_label_2d} [V]")
        self.ax1.set_ylabel(f"{self.y_label_2d} [V]")
        self.ax2.set_xlabel(f"{self.x_label_2d} [V]")
        self.ax2.set_ylabel(f"{self.z_label_2d} [uA]")
        
        X_num = int(round(abs(X_end_voltage - X_start_voltage) / abs(X_step))) + 1
        Y_num = int(round(abs(Y_end_voltage - Y_start_voltage) / abs(Y_step))) + 1
        data = np.zeros((Y_num, X_num))
        self.im = self.ax1.imshow(data, cmap='bwr', aspect='auto', origin='lower',
                                extent=[X_start_voltage, X_end_voltage, Y_start_voltage, Y_end_voltage])
        data_matrix = []
        plt.colorbar(self.im, ax=self.ax1, label=r"I [uA]")
        while True:
            initial_state = initial_state_basic.copy()
            self.Y_voltage = Y_voltage
            for Y_gate in Y_swept_outputs.gates:
                initial_state.append([Y_gate, Y_voltage])
            params['initial_state'] = initial_state
            params['ax2'] = self.ax2  
            params['is_2d'] = True   
            X_values, Z_values = self.sweep1D(**params)
            
            
            data[loop] = Z_values
            data_matrix.append(Z_values)
            self.im.set_data(data)
            self.im.set_clim(vmin=np.nanmin(data), vmax=np.nanmax(data))
            self.fig.canvas.draw_idle()
            
            loop += 1
                
            #data = np.loadtxt(self.filename+'.txt', skiprows=1, usecols=1)
            
            #data_matrix.append(data)
            if (Y_start_voltage < Y_end_voltage and Y_voltage > Y_end_voltage - 1e-6) or (
                    Y_start_voltage > Y_end_voltage and Y_voltage < Y_end_voltage + 1e-6):
                break
            Y_voltage = Y_start_voltage + loop * Y_step if Y_start_voltage < Y_end_voltage else Y_start_voltage - loop * Y_step
            
        plt.ioff()
        plt.close()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(data, aspect='auto', cmap='bwr', origin='lower',
                       extent=[min(X_start_voltage, X_end_voltage), max(X_start_voltage, X_end_voltage),
                               min(Y_start_voltage, Y_end_voltage), max(Y_start_voltage, Y_end_voltage)])
        plt.colorbar(im, ax=ax, label=r"I [uA]")
        plt.title(f"2D Sweep: [{self.z_label_2d}] vs [{self.x_label_2d} & {self.y_label_2d}]")
        ax.set_xlabel(self.set_gates_group_label(X_swept_outputs)+'[V]')
        ax.set_ylabel(self.set_gates_group_label(Y_swept_outputs)+'[V]')
        plt.savefig(f"{self.filename_2d}.png", dpi=1000)
        plt.close()


    def sweepTime(self, measured_inputs: GatesGroup, total_time: float,
                time_step: float, initial_state: list = None, comments: str = None) -> None:
        """
        Perform a 1D voltage sweep and create an animated plot of the measurement.

        Args:L
            swept_outputs (GatesGroup): Group of output gates to sweep.
            measured_inputs (GatesGroup): Group of input gate for measuring current.
            start_voltage (float): Starting voltage for sweep.
            end_voltage (float): End voltage for sweep.
            step (float): Increment for each voltage step.
            initial_state (list): List of initial voltages for gates.
            comments (str): Additional comments for logging.
        """
        self.x_label = 'time'
        self.y_label = self.set_gates_group_label(measured_inputs)
        self.y_labels = [self.set_gate_label(measured_input) for measured_input in measured_inputs.gates]
        self.comments = comments
        self.set_filenema()

        self.total_time = total_time
        self.time_step = time_step

        # Progress bar for ramping up
        pbar = tqdm(total=len(self.outputs.gates), desc="[INFO] Ramping voltage", ncols=80,
                    leave=True)

        idle_gates = []
        for gate in self.outputs.gates:
            if gate not in [state[0] for state in initial_state]:
                idle_gates.append(gate)
        GatesGroup(idle_gates).turn_off()
        pbar.update(len(idle_gates))

        # Set initial voltage for each gate in initial_state
        for gate, initial_voltage in initial_state:
            gate.voltage(initial_voltage, False)

        # Wait for initial voltages to stabilize
        while not all([gate.is_at_target_voltage(voltage) for gate, voltage in initial_state]):
            time.sleep(0.1)
        pbar.update(len(initial_state))
        pbar.close()
        time.sleep(1)

        # Set up plot
        num = len(measured_inputs.gates)
        fig, axes = plt.subplots(num, 1, figsize=(8, 5*num))
        if isinstance(axes, np.ndarray):
            lines = [None for _ in range(num)]
            for i, ax in enumerate(axes.flat):
                lines[i], = ax.plot([], [])
                ax.set_xlabel(f"{self.x_label} [s]")
                ax.set_ylabel(f"{self.y_labels[i]} [uA]")
        else:
            lines, = axes.plot([], [])
            axes.set_xlabel(f"{self.x_label} [s]")
            axes.set_ylabel(f"{self.y_label} [uA]")

        self.times = []
        self.currents = [[] for _ in range(num)]


        # Record parameters
        self.log_params("time")

        # Start data collection
        print("[INFO] Start recording.")
        with open(f"{self.filename}.txt", 'a') as file:
            file.write(f"{self.x_label + ' [s]':>24}"
                       + "".join(f"{label + ' [uA]':>24}" for label in self.y_labels)
                       + "\n"
                       )

        # Execute sweep and record data
        total = self.total_time // self.time_step
        pbar = tqdm(total=total, desc="[INFO] Sweeping", ncols=80, leave=True)  # progress bar
        frame = 0
        initial_time = time.time()
        current_time_list = []
        while True:
            current = [None for _ in range(num)]
            self.current_time = time.time() - initial_time
            current_time_list.append(self.current_time)
            if isinstance(axes, np.ndarray):
                for i, measured_input in enumerate(measured_inputs.gates):
                    current[i] = measured_input.read_current(self.amplification)
                    self.currents[i].append(current[i])
                    # Update plot limits and data
                    if len(self.currents[0]) > 1:
                        axes[i].set_xlim(0.0, self.current_time + self.time_step)
                        axes[i].set_ylim(min(self.currents[i]) - (max(self.currents[i])-min(self.currents[i]))/3,
                                         max(self.currents[i]) + (max(self.currents[i])-min(self.currents[i]))/3)
                    lines[i].set_data(current_time_list, self.currents[i])
            else:
                current[0] = measured_inputs.gates[0].read_current(self.amplification)
                self.currents[0].append(current[0])
                if len(self.currents[0]) > 1:
                    axes.set_xlim(0.0, self.current_time + self.time_step)
                    axes.set_ylim(min(self.currents[0]) - (max(self.currents[0])-min(self.currents[0]))/3,
                                  max(self.currents[0]) + (max(self.currents[0])-min(self.currents[0]))/3)
                lines.set_data(current_time_list, self.currents[0])

            plt.draw()
            plt.pause(0.1)
            frame += 1
            pbar.update(1)

            with open(f"{self.filename}.txt", 'a') as file:
                file.write(f"{self.current_time:>24.2f}"
                           + "".join(f"{curr:>24.16f}" for curr in current)
                           + " \n"
                           )
            
            while True:
                if time.time() - initial_time > current_time_list[-1] + self.time_step:
                    break
                time.sleep(self.time_step / 100)

            if (self.current_time >= self.total_time):
                pbar.close()
                break

        plt.savefig(f"{self.filename}.png", dpi=300)
        print("[INFO] Data collection complete and figure saved. \n")
