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
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from conductorquantum import ConductorQuantum

from .gate import GatesGroup, Gate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class Sweeper:
    """
    Sweeper class to perform and log voltage sweeps on defined gates.
    """

    def __init__(self, outputs: GatesGroup = None, inputs: GatesGroup = None, 
                 temperature: str = None, device: str = None) -> None:
        """Initialize the Sweeper class.
        
        Args:
            outputs (GatesGroup): Group of output gates to control
            inputs (GatesGroup): Group of input gates to measure
            temperature (str): Temperature of the experiment
            device (str): Device identifier
        
        Raises:
            ValueError: If outputs or inputs are not GatesGroup instances
        """
        if outputs is not None and not isinstance(outputs, GatesGroup):
            raise ValueError("outputs must be a GatesGroup instance")
        if inputs is not None and not isinstance(inputs, GatesGroup):
            raise ValueError("inputs must be a GatesGroup instance")
            
        self.outputs = outputs
        self.inputs = inputs
        self.temperature = temperature
        self.device = device

        # Create necessary directories
        try:
            os.makedirs("data", exist_ok=True)
            os.makedirs("figures", exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create directories: {str(e)}")
            raise

        # Initialize other attributes
        self._initialize_attributes()
        
        logger.info("Sweeper initialized successfully")
        
    def _initialize_attributes(self):
        """Initialize class attributes with default values."""
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
            base_filename = f"{date.today().strftime('%Y%m%d')}_{self.temperature}_[{self.z_label}]_vs_time"
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
                if not self.is_2d_sweep:
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

    def _validate_voltage_params(self, start_voltage: list, end_voltage: list, step: list) -> None:
        """Validate voltage sweep parameters.
        
        Args:
            start_voltage (list): Starting voltage as [value, unit]
            end_voltage (list): Ending voltage as [value, unit]
            step (list): Step size as [value, unit]
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not all(isinstance(v, list) and len(v) == 2 for v in [start_voltage, end_voltage, step]):
            raise ValueError("Voltage parameters must be lists of [value, unit]")
            
        if step[0] <= 0:
            raise ValueError("Step size must be positive")
            
        # Convert to base units for comparison
        start_v = self._convert_units(start_voltage)
        end_v = self._convert_units(end_voltage)
        step_v = self._convert_units(step)
        
        if abs(end_v - start_v) < step_v:
            raise ValueError("Step size is larger than voltage range")
            
    def _validate_units(self, unit: str, unit_type: str = 'voltage') -> None:
        """Validate unit specifications.
        
        Args:
            unit (str): Unit to validate
            unit_type (str): Type of unit ('voltage' or 'current')
            
        Raises:
            ValueError: If unit is invalid
        """
        voltage_units = {'V', 'mV', 'uV', 'nV'}
        current_units = {'mA', 'uA', 'nA', 'pA'}
        
        if unit_type == 'voltage' and unit not in voltage_units:
            raise ValueError(f"Invalid voltage unit. Must be one of {voltage_units}")
        elif unit_type == 'current' and unit not in current_units:
            raise ValueError(f"Invalid current unit. Must be one of {current_units}")

    def _validate_model(self, model: str, sweep_type: str) -> None:
        """Validate that the provided model is supported for the given sweep type.
        
        Args:
            model (str): Name of the model to validate
            sweep_type (str): Type of sweep ('1D', '2D', or 'time')
            
        Raises:
            ValueError: If the model is not supported for the sweep type
        """
        if model is None:
            return
            
        supported_models = {
            '1D': {
                'pinch-off-classifier',
                'pinch-off-parameter-extractor',
                'turn-on-classifier',
                'turn-on-parameter-extractor',
                'coulomb-blockade-classifier',
                'coulomb-blockade-peak-detector'
            },
            '2D': {
                'charge-stability-diagram-classifier',
                'charge-stability-diagram-segmenter',
                'triple-point-detector',
                'honeycomb-pattern-detector'
            },
            'time': {
                'noise-analyzer',
                'drift-detector',
                'stability-analyzer'
            }
        }
        
        if sweep_type not in supported_models:
            raise ValueError(f"Invalid sweep type: {sweep_type}")
            
        if model not in supported_models[sweep_type]:
            raise ValueError(
                f"Model '{model}' is not supported for {sweep_type} sweeps. "
                f"Supported models are: {sorted(supported_models[sweep_type])}"
            )
            
    def sweep1D(self, 
                swept_outputs: GatesGroup, 
                measured_inputs: GatesGroup, 
                start_voltage: list[float, str], 
                end_voltage: list[float, str],
                step: list[float, str], 
                initial_state: list = [], 
                voltage_unit: str = 'V',
                current_unit: str = 'uA',
                model: str = None,
                comments: str = None, 
                ax2=None, 
                is_2d_sweep: bool = False,
                is_show: bool = True
                ) -> tuple:
        """
        Perform a 1D voltage sweep and generate an animated plot.
        
        Args:
            swept_outputs (GatesGroup): Gates to sweep
            measured_inputs (GatesGroup): Gates to measure
            start_voltage (list): Starting voltage as [value, unit]
            end_voltage (list): Ending voltage as [value, unit]
            step (list): Step size as [value, unit]
            initial_state (list): Initial state of the gates
            voltage_unit (str): Unit for voltage measurements
            current_unit (str): Unit for current measurements
            model (str): Model name for analysis. Supported models:
                - 'pinch-off-classifier': Classifies pinch-off curves
                - 'pinch-off-parameter-extractor': Extracts parameters from pinch-off curves
                - 'turn-on-classifier': Classifies turn-on curves
                - 'turn-on-parameter-extractor': Extracts parameters from turn-on curves
                - 'coulomb-blockade-classifier': Classifies Coulomb blockade patterns
                - 'coulomb-blockade-peak-detector': Detects peaks in Coulomb blockade measurements
            comments (str): Comments for the run
            ax2 (Axes): Axes for the plot
            is_2d_sweep (bool): Whether to perform a 2D sweep
            is_show (bool): Whether to show the plot after completion

        Raises:
            ValueError: If input parameters are invalid.

        Returns:
            tuple: Tuple containing the voltage and current arrays.
            None: If the plot is not shown.
        """
        try:
            # Validate model if provided
            self._validate_model(model, '1D')
            
            # Validate inputs
            self._validate_voltage_params(start_voltage, end_voltage, step)
            self._validate_units(voltage_unit, 'voltage')
            self._validate_units(current_unit, 'current')
            
            # Set sweep labels and units
            self.x_label = self._set_gates_group_label(swept_outputs)
            self.z_label = self._set_gates_group_label(measured_inputs)
            self.voltage_unit = voltage_unit
            self.current_unit = current_unit
            self.comments = comments
            self.model = model
            self.ax2 = ax2
            self.is_2d_sweep = is_2d_sweep
            self.is_show = is_show
            
            self._set_units()
            
            if not self.is_2d_sweep:
                self._set_filename('1D')

            # Convert voltage parameters
            self.X_start_voltage = self._convert_units(start_voltage)
            self.X_end_voltage = self._convert_units(end_voltage)
            self.X_step = self._convert_units(step)
            
            # Pre-allocate data arrays for better performance
            total_steps = round(abs(self.X_end_voltage - self.X_start_voltage) / self.X_step + 1)
            self.X_voltages = np.zeros(total_steps)
            self.currents = np.zeros(total_steps)
            
            # Initialize plotting
            if self.ax2 is None:
                plt.ion()
                fig, self.ax2 = plt.subplots(1, 1, figsize=(12, 7))
                self._setup_plot_style(fig)
            else:
                self.ax2.clear()
                if self.Y_voltage is not None:
                    self.ax2.set_title(f"{self.y_label}: {self.convert_si_value(self.Y_voltage, 'V')}")
                    
            self._setup_axes_labels()
            self.lines, = self.ax2.plot([], [], 'b-', lw=2)
            
            # Set initial state
            self._set_initial_state(initial_state, swept_outputs)
            
            # Log parameters and start sweep
            self._log_params(sweep_type='voltage', status='start')
            self._write_data_header()
            
            logger.info(f"Starting 1D sweep from {self.X_start_voltage*self.voltage_scale:.3f} "
                       f"[{self.voltage_unit}] to {self.X_end_voltage*self.voltage_scale:.3f} "
                       f"[{self.voltage_unit}]")
            
            # Perform sweep
            for i in tqdm(range(total_steps), desc="Sweeping", ncols=80):
                self.X_voltage = (self.X_start_voltage + i * self.X_step 
                                if self.X_start_voltage < self.X_end_voltage 
                                else self.X_start_voltage - i * self.X_step)
                
                # Set voltage and measure
                swept_outputs.voltage(self.X_voltage)
                self.X_voltages[i] = self.X_voltage * self.voltage_scale
                self.currents[i] = measured_inputs.gates[0].read_current() * self.current_scale
                
                # Update plot
                self._update_plot(i)
                self._write_measurement_data(i)
                
            # Finalize
            if self.is_2d_sweep:
                logger.info("1D sweep complete")
                return self.X_voltages, self.currents
            else:
                self._save_and_show_plot()
                logger.info("1D sweep complete and figure saved")
                self._log_params(sweep_type='voltage', status='end')
                
        except Exception as e:
            logger.error(f"Error during 1D sweep: {str(e)}")
            raise
            
    def _setup_plot_style(self, fig):
        """Set up the plot style."""
        plt.rc('legend', fontsize=22, framealpha=0.9)
        plt.rc('xtick', labelsize=24, color='#2C3E50')
        plt.rc('ytick', labelsize=24, color='#2C3E50')
        fig.patch.set_facecolor('white')
        
    def _setup_axes_labels(self):
        """Set up the axes labels."""
        self.ax2.set_xlabel(f"{self.x_label} [{self.voltage_unit}]",
                           color='#2C3E50', fontsize=32)
        self.ax2.set_ylabel(f"{self.z_label} [{self.current_unit}]",
                           color='#2C3E50', fontsize=32)
                           
    def _set_initial_state(self, initial_state, swept_outputs):
        """Set up initial state for the sweep."""
        with tqdm(total=len(initial_state)+len(swept_outputs.gates),
                 desc="Ramping voltage", ncols=80) as pbar:
            # Set initial states
            converted_init_state = []
            for gate, init_volt, init_unit in initial_state:
                converted_init_volt = self._convert_units([init_volt, init_unit])
                converted_init_state.append([gate, converted_init_volt])
                gate.voltage(converted_init_volt, is_wait=False)
            
            # Wait for stabilization
            while not all([gate.is_at_target_voltage(voltage) 
                         for gate, voltage in converted_init_state]):
                time.sleep(0.1)
            pbar.update(len(initial_state))
            
            # Set swept outputs
            swept_outputs.voltage(self.X_start_voltage)
            pbar.update(len(swept_outputs.gates))
            
        time.sleep(0.1)
        
    def _update_plot(self, index):
        """Update the plot with new data."""
        if index > 0:  # Only update after first point
            self.ax2.set_xlim(
                min(self.X_voltages[:index+1]) - self.X_step * self.voltage_scale,
                max(self.X_voltages[:index+1]) + self.X_step * self.voltage_scale
            )
            curr_min = min(self.currents[:index+1])
            curr_max = max(self.currents[:index+1])
            if curr_min == curr_max:
                curr_min -= 0.001
                curr_max += 0.001
            self.ax2.set_ylim(
                curr_min - (curr_max - curr_min) / 4,
                curr_max + (curr_max - curr_min) / 4
            )
        self.lines.set_data(self.X_voltages[:index+1], self.currents[:index+1])
        plt.draw()
        plt.pause(0.01)
        
    def _write_data_header(self):
        """Write the data file header."""
        if not self.is_2d_sweep:
            try:
                with open(f"data/{self.filename}.txt", 'a') as file:
                    header = (f"{self.x_label} [{self.voltage_unit}]".rjust(16) +
                            f"{self.z_label} [{self.current_unit}]".rjust(16))
                    file.write(header + "\n")
            except IOError as e:
                logger.error(f"Failed to write data header: {str(e)}")
                raise
                
    def _write_measurement_data(self, index):
        """Write measurement data to file."""
        try:
            with open(f"data/{self.filename}.txt", 'a') as file:
                if self.is_2d_sweep:
                    file.write(f"{self.Y_voltage * self.voltage_scale:>16.4f} "
                             f"{self.X_voltages[index]:>16.4f} "
                             f"{self.currents[index]:>16.8f}\n")
                else:
                    file.write(f"{self.X_voltages[index]:>16.4f} "
                             f"{self.currents[index]:>16.8f}\n")
        except IOError as e:
            logger.error(f"Failed to write measurement data: {str(e)}")
            raise
            
    def _save_and_show_plot(self):
        """Save and optionally show the final plot."""
        try:
            plt.ioff()
            plt.tight_layout()
            plt.savefig(f"figures/{self.filename}.png", dpi=300, bbox_inches='tight')
            if self.is_show:
                plt.show()
            else:
                plt.close()
        except Exception as e:
            logger.error(f"Failed to save or show plot: {str(e)}")
            raise

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
                model: str = None,
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
            model (str): Model name for analysis. Supported models:
                - 'charge-stability-diagram-classifier': Classifies charge stability diagrams
                - 'charge-stability-diagram-segmenter': Segments regions in charge stability diagrams
                - 'triple-point-detector': Detects triple points in stability diagrams
                - 'honeycomb-pattern-detector': Detects honeycomb patterns in stability diagrams
            comments (str): Additional comments for logging.
            is_show (bool): Whether to show the plot after completion.
        """
        try:
            # Validate model if provided
            self._validate_model(model, '2D')
            
            # Validate inputs
            for params in [(X_start_voltage, X_end_voltage, X_step), 
                          (Y_start_voltage, Y_end_voltage, Y_step)]:
                self._validate_voltage_params(*params)
            self._validate_units(voltage_unit, 'voltage')
            self._validate_units(current_unit, 'current')
            
            # Set up sweep parameters
            self.voltage_unit = voltage_unit
            self.current_unit = current_unit
            self.is_2d_sweep = True
            self._set_units()
            
            # Convert voltage parameters
            self.X_start_voltage = self._convert_units(X_start_voltage)
            self.X_end_voltage = self._convert_units(X_end_voltage)
            self.X_step = self._convert_units(X_step)
            self.Y_start_voltage = self._convert_units(Y_start_voltage)
            self.Y_end_voltage = self._convert_units(Y_end_voltage)
            self.Y_step = self._convert_units(Y_step)
            
            # Set labels and filename
            self.x_label = self._set_gates_group_label(X_swept_outputs)
            self.y_label = self._set_gates_group_label(Y_swept_outputs)
            self.z_label = self._set_gates_group_label(measured_inputs)
            self.comments = comments
            self._set_filename('2D')
            
            # Calculate array dimensions
            X_num = int(round(abs(self.X_end_voltage - self.X_start_voltage) / self.X_step)) + 1
            Y_num = int(round(abs(self.Y_end_voltage - self.Y_start_voltage) / self.Y_step)) + 1
            
            # Pre-allocate data array
            data = np.full((Y_num, X_num), np.nan)
            
            # Write header and start logging
            self._write_2d_data_header()
            self._log_params(sweep_type='voltage', status='start')
            
            # Set up plotting
            plt.ion()
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
            self._setup_2d_plot(fig, ax1, ax2, data)
            
            # Prepare 1D sweep parameters
            params = {
                'swept_outputs': X_swept_outputs,
                'start_voltage': X_start_voltage,
                'end_voltage': X_end_voltage,
                'step': X_step,
                'measured_inputs': measured_inputs,
                'initial_state': initial_state.copy(),
                'voltage_unit': voltage_unit,
                'current_unit': current_unit,
                'comments': comments,
                'is_2d_sweep': True,
                'is_show': False
            }
            
            # Perform 2D sweep
            logger.info(f"Starting 2D sweep with {Y_num} Y steps and {X_num} X steps per Y value")
            with tqdm(total=Y_num, desc="Y-axis sweep progress", ncols=80) as pbar:
                self.Y_voltage = self.Y_start_voltage
                for idx in range(Y_num):
                    # Update Y voltage in initial state
                    current_initial_state = initial_state.copy()
                    for Y_gate in Y_swept_outputs.gates:
                        current_initial_state.append([Y_gate, self.Y_voltage, 'V'])
                    params['initial_state'] = current_initial_state
                    params['ax2'] = ax2
                    
                    # Perform 1D sweep
                    _, Z_values = self.sweep1D(**params)
                    data[idx] = Z_values
                    
                    # Update plot
                    self._update_2d_plot(data, idx)
                    
                    # Calculate next Y voltage
                    if idx < Y_num - 1:
                        self.Y_voltage += self.Y_step if self.Y_start_voltage < self.Y_end_voltage else -self.Y_step
                    pbar.update(1)
            
            # Finalize and save
            plt.ioff()
            plt.close('all')
            self._log_params(sweep_type='voltage', status='end')
            
            # Generate and save final plot
            self._generate_final_2d_plot(data, is_show)
            logger.info("2D sweep completed successfully")
            
        except Exception as e:
            logger.error(f"Error during 2D sweep: {str(e)}")
            raise
            
    def _write_2d_data_header(self):
        """Write header for 2D sweep data file."""
        try:
            with open(f"data/{self.filename}.txt", 'a') as file:
                header = (f"{self.y_label} [{self.voltage_unit}]".rjust(16) +
                         f"{self.x_label} [{self.voltage_unit}]".rjust(16) +
                         f"{self.z_label} [{self.current_unit}]".rjust(16))
                file.write(header + "\n")
        except IOError as e:
            logger.error(f"Failed to write data header: {str(e)}")
            raise
            
    def _setup_2d_plot(self, fig, ax1, ax2, data):
        """Set up the initial 2D plot."""
        # Configure axes
        ax1.set_xlabel(f"{self.x_label} [{self.voltage_unit}]", fontsize=12)
        ax1.set_ylabel(f"{self.y_label} [{self.voltage_unit}]", fontsize=12)
        ax2.set_xlabel(f"{self.x_label} [{self.voltage_unit}]", fontsize=12)
        ax2.set_ylabel(f"{self.z_label} [{self.current_unit}]", fontsize=12)
        
        # Set up colormap
        colorsbar = ['#02507d', '#ede8e5', '#b5283b']
        cm = LinearSegmentedColormap.from_list('', colorsbar, N=500)
        
        # Create image plot
        self.img = ax1.imshow(
            data, cmap=cm, aspect='auto', origin='lower',
            extent=[self.X_start_voltage, self.X_end_voltage, 
                   self.Y_start_voltage, self.Y_end_voltage],
            interpolation='none'
        )
        
        # Configure figure
        fig.patch.set_facecolor('white')
        
        # Add colorbar
        self.cbar = fig.colorbar(self.img, ax=ax1, pad=0.005, extend='both')
        self.cbar.ax.set_title(f'{self.z_label} [{self.current_unit}]', pad=10)
        self.cbar.ax.tick_params(direction='in', width=2, length=5, labelsize=12)
        
    def _update_2d_plot(self, data, current_idx):
        """Update the 2D plot with new data."""
        self.img.set_data(data)
        
        # Update colorbar limits
        valid_data = data[np.isfinite(data)]
        if len(valid_data) > 0:
            clim_min = np.nanmin(valid_data)
            clim_max = np.nanmax(valid_data)
            self.img.set_clim(clim_min, clim_max)
            
            # Update colorbar ticks
            barticks = np.linspace(clim_min, clim_max, 5)
            self.cbar.set_ticks(barticks)
            self.cbar.ax.set_yticklabels([f"{t:.2f}" for t in barticks])
            self.cbar.update_normal(self.img)
            
        plt.draw()
        plt.pause(0.01)
        
    def _generate_final_2d_plot(self, data, is_show):
        """Generate and save the final 2D plot."""
        try:
            # Set up colormap
            colorsbar = ['#02507d', '#ede8e5', '#b5283b']
            cm = LinearSegmentedColormap.from_list('', colorsbar, N=500)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 7))
            img = ax.imshow(
                data, vmin=data.min(), vmax=data.max(),
                cmap=cm, aspect='auto', origin='lower',
                extent=[self.X_start_voltage * self.voltage_scale, self.X_end_voltage * self.voltage_scale,
                       self.Y_start_voltage * self.voltage_scale, self.Y_end_voltage * self.voltage_scale],
                interpolation='none'
            )
            
            # Configure plot style
            plt.rc('legend', fontsize=22, framealpha=0.9)
            plt.rc('xtick', labelsize=24, color='#2C3E50')
            plt.rc('ytick', labelsize=24, color='#2C3E50')
            fig.patch.set_facecolor('white')
            
            # Add and configure colorbar
            barticks = np.linspace(data.min(), data.max(), 5)
            barticks = np.around(barticks, 2)
            cbar = fig.colorbar(img, pad=0.005, extend='both')
            cbar.set_ticks(barticks)
            cbar.ax.set_yticklabels(barticks)
            cbar.ax.set_title(f'{self.z_label} [{self.current_unit}]',
                            fontsize=28, pad=10)
            cbar.ax.tick_params(direction='in', width=2, length=5, labelsize=22)
            
            # Configure axes
            for spine in ax.spines.values():
                spine.set_color('#2C3E50')
            
            ax.set_xlabel(f"{self.x_label} [{self.voltage_unit}]",
                         color='#2C3E50', fontsize=32)
            ax.set_ylabel(f"{self.y_label} [{self.voltage_unit}]",
                         color='#2C3E50', fontsize=32)
            
            ax.tick_params(axis='both', direction='in', width=4,
                         length=10, pad=10, labelsize=24)
            ax.tick_params(axis='x', top=False)
            
            # Save and show plot
            plt.tight_layout()
            plt.savefig(f"figures/{self.filename}.png", dpi=300,
                       bbox_inches='tight')
            logger.info("2D plot saved successfully")
            
            if is_show:
                plt.show()
                
        except Exception as e:
            logger.error(f"Failed to generate final 2D plot: {str(e)}")
            raise

    def sweepTime(self, 
                  measured_inputs: GatesGroup, 
                  total_time: float,
                  time_step: float, 
                  initial_state: list, 
                  current_unit: str = 'uA',
                  model: str = None,
                  comments: str = None,
                  is_show: bool = True
                  ) -> None:
        """
        Perform a time-based sweep by recording current measurements over a specified duration.

        Args:
            measured_inputs (GatesGroup): Group of input gates for measurement.
            total_time (float): Total duration of the sweep in seconds.
            time_step (float): Time interval between measurements in seconds.
            initial_state (list): List of tuples (gate, init_voltage) for initial state.
            current_unit (str): Unit for current measurements.
            model (str): Model name for analysis. Supported models:
                - 'noise-analyzer': Analyzes noise characteristics in time series data
                - 'drift-detector': Detects and characterizes drift in measurements
                - 'stability-analyzer': Analyzes stability of measurements over time
            comments (str): Additional comments for logging.
            is_show (bool): Whether to show the plot after completion.
            
        Raises:
            ValueError: If input parameters are invalid.
        """
        try:
            # Validate model if provided
            self._validate_model(model, 'time')
            
            # Validate inputs
            if total_time <= 0:
                raise ValueError("Total time must be positive")
            if time_step <= 0:
                raise ValueError("Time step must be positive")
            if time_step >= total_time:
                raise ValueError("Time step must be smaller than total time")
            self._validate_units(current_unit, 'current')
            
            # Set up parameters
            self.x_label = 'time'
            self.z_label = self._set_gates_group_label(measured_inputs)
            self.current_unit = current_unit
            self.comments = comments
            self.total_time = total_time
            self.time_step = time_step
            
            self._set_units()
            self._set_filename('time')
            
            # Pre-allocate arrays
            total_steps = int(total_time // time_step) + 1
            time_points = np.zeros(total_steps)
            currents = np.zeros(total_steps)
            
            # Set up initial state
            logger.info("Setting up initial state")
            self._setup_time_sweep_initial_state(initial_state)
            
            # Set up plotting
            fig, ax = plt.subplots(figsize=(12, 7))
            self._setup_time_sweep_plot(ax)
            lines, = ax.plot([], [], 'b-', lw=2)
            
            # Write header and start logging
            self._write_time_sweep_header()
            self._log_params(sweep_type='time', status='start')
            
            # Perform time sweep
            logger.info(f"Starting time sweep for {total_time:.1f}s with {time_step:.3f}s steps")
            initial_time = time.time()
            
            with tqdm(total=total_steps, desc="Recording measurements", ncols=80) as pbar:
                for i in range(total_steps):
                    # Record time and current
                    current_time = time.time() - initial_time
                    time_points[i] = current_time
                    currents[i] = measured_inputs.gates[0].read_current() * self.current_scale
                    
                    # Update plot
                    self._update_time_sweep_plot(ax, lines, time_points, currents, i)
                    
                    # Write data
                    self._write_time_sweep_data(current_time, currents[i])
                    
                    # Wait for next measurement
                    next_measurement_time = initial_time + (i + 1) * time_step
                    while time.time() < next_measurement_time:
                        time.sleep(time_step / 100)
                        
                    pbar.update(1)
            
            # Finalize and save
            self._save_time_sweep_plot(fig, is_show)
            self._log_params(sweep_type='time', status='end')
            logger.info("Time sweep completed successfully")
            
        except Exception as e:
            logger.error(f"Error during time sweep: {str(e)}")
            raise
            
    def _setup_time_sweep_initial_state(self, initial_state):
        """Set up the initial state for time sweep."""
        try:
            # Turn off gates not in initial state
            idle_gates = [gate for gate in self.outputs.gates 
                         if gate not in [state[0] for state in initial_state]]
            
            with tqdm(total=len(initial_state) + len(idle_gates),
                     desc="Setting voltages", ncols=80) as pbar:
                # Turn off unused gates
                if idle_gates:
                    GatesGroup(idle_gates).turn_off()
                    pbar.update(len(idle_gates))
                
                # Set initial voltages
                converted_init_state = []
                for gate, init_volt, init_unit in initial_state:
                    converted_volt = self._convert_units([init_volt, init_unit])
                    converted_init_state.append([gate, converted_volt])
                    gate.voltage(converted_volt, is_wait=False)
                
                # Wait for voltage stabilization
                while not all(gate.is_at_target_voltage(voltage) 
                            for gate, voltage in converted_init_state):
                    time.sleep(0.1)
                pbar.update(len(initial_state))
                
            time.sleep(0.1)  # Final stabilization
            
        except Exception as e:
            logger.error(f"Failed to set initial state: {str(e)}")
            raise
            
    def _setup_time_sweep_plot(self, ax):
        """Set up the plot for time sweep."""
        ax.set_xlabel(f"{self.x_label} [s]", color='#2C3E50', fontsize=32)
        ax.set_ylabel(f"{self.z_label} [{self.current_unit}]",
                     color='#2C3E50', fontsize=32)
        ax.tick_params(axis='both', direction='in', width=4,
                      length=10, pad=10, labelsize=24)
        for spine in ax.spines.values():
            spine.set_color('#2C3E50')
            
    def _write_time_sweep_header(self):
        """Write header for time sweep data file."""
        try:
            with open(f"data/{self.filename}.txt", 'a') as file:
                header = (f"{self.x_label} [s]".rjust(16) + 
                         f"{self.z_label} [{self.current_unit}]".rjust(16))
                file.write(header + "\n")
        except IOError as e:
            logger.error(f"Failed to write time sweep header: {str(e)}")
            raise
            
    def _update_time_sweep_plot(self, ax, lines, time_points, currents, current_idx):
        """Update the plot during time sweep."""
        if current_idx > 0:
            # Update axis limits
            ax.set_xlim(0, time_points[current_idx] + self.time_step)
            curr_min = min(currents[:current_idx+1])
            curr_max = max(currents[:current_idx+1])
            if curr_min == curr_max:
                curr_min -= 0.001
                curr_max += 0.001
            ax.set_ylim(curr_min - (curr_max - curr_min) / 4,
                       curr_max + (curr_max - curr_min) / 4)
                       
        # Update plot data
        lines.set_data(time_points[:current_idx+1], currents[:current_idx+1])
        plt.draw()
        plt.pause(0.01)
        
    def _write_time_sweep_data(self, current_time, current):
        """Write measurement data during time sweep."""
        try:
            with open(f"data/{self.filename}.txt", 'a') as file:
                file.write(f"{current_time:>16.2f} {current:>16.8f}\n")
        except IOError as e:
            logger.error(f"Failed to write time sweep data: {str(e)}")
            raise
            
    def _save_time_sweep_plot(self, fig, is_show):
        """Save and optionally display the final time sweep plot."""
        try:
            plt.ioff()
            plt.tight_layout()
            plt.savefig(f"figures/{self.filename}.png", dpi=300,
                       bbox_inches='tight')
            logger.info("Time sweep plot saved successfully")
            
            if is_show:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            logger.error(f"Failed to save time sweep plot: {str(e)}")
            raise

    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
        
    def cleanup(self):
        """Clean up resources and reset gates."""
        try:
            # Close all matplotlib figures
            plt.close('all')
            
            # Reset all outputs to 0V if they exist
            if self.outputs:
                self.outputs.turn_off()
                
            # Reset attributes
            self._initialize_attributes()
            
        except Exception as e:
            print(f"Warning: Cleanup encountered an error: {str(e)}")