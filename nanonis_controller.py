# -*- coding: utf-8 -*-
"""
This script performs voltage sweep experiments using a Nanonis instance.

"""

from nanonis_spm import Nanonis
import socket
import time
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import os
from datetime import datetime
from GateClass import Gate
from decimal import Decimal

# Create a socket connection to Nanonis
connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
connection.connect(("192.168.236.1", 6501))

# Create Nanonis instance for controlling the device
nanonisInstance = Nanonis(connection)

# %% Define gates
#  The following gates represent the outputs and inputs for voltage control.

# Define output gates
t_P1 = Gate(name='output1', label='t_P1', read_index=24, write_index=1, nanonisInstance=nanonisInstance)
t_P2 = Gate(name='output2', label='t_P2', read_index=25, write_index=2, nanonisInstance=nanonisInstance)
t_P3 = Gate(name='output3', label='t_P3', read_index=26, write_index=3, nanonisInstance=nanonisInstance)
t_P4 = Gate(name='output4', label='t_P4', read_index=27, write_index=4, nanonisInstance=nanonisInstance)
t_barriers = Gate(name='output5', label='t_barriers', read_index=28, write_index=5, nanonisInstance=nanonisInstance)
res_S_D = Gate(name='output6', label='res_S_D', read_index=29, write_index=6, nanonisInstance=nanonisInstance)
t_s = Gate(name='output7', label='t_s', read_index=30, write_index=7, nanonisInstance=nanonisInstance)
G8 = Gate(name='output8', label='G8', read_index=31, write_index=8, nanonisInstance=nanonisInstance)

output_gates = [t_P1, t_P2, t_P3, t_P4, t_barriers, res_S_D, t_s, G8]

# Define input gates
t_D = Gate(name='input1', label='t_D', read_index=0, nanonisInstance=nanonisInstance)
b_D = Gate(name='input1', label='b_D', read_index=1, nanonisInstance=nanonisInstance)
SD3 = Gate(name='input1', label='SD3', read_index=2, nanonisInstance=nanonisInstance)
SD4 = Gate(name='input1', label='SD4', read_index=3, nanonisInstance=nanonisInstance)
SD5 = Gate(name='input1', label='SD5', read_index=4, nanonisInstance=nanonisInstance)
SD6 = Gate(name='input1', label='SD6', read_index=5, nanonisInstance=nanonisInstance)
SD7 = Gate(name='input1', label='SD7', read_index=6, nanonisInstance=nanonisInstance)
SD8 = Gate(name='input1', label='SD8', read_index=7, nanonisInstance=nanonisInstance)

input_gates = [t_D, b_D, SD3, SD4, SD5, SD6, SD7, SD8]

# %% Set all to 0
value_to_set = 0

for output_gate in output_gates:
    output_gate.voltage(value_to_set)


# %% Log file

def log_params(gates_out: list[Gate], gate_in: Gate, start_voltage: float, end_voltage: float, step: float,
               filename: str) -> None:
    """
    Log the parameters of the sweep to a file.

    Args:
        gates_out (list[Gate]): List of output gates that are being swept.
        gate_in (Gate): The input gate whose current is measured.
        start_voltage (float): Starting voltage of the sweep.
        end_voltage (float): End voltage of the sweep.
        step (float): Step size for the sweep.
        filename (str): Name of the file where results are saved.

    Returns:
        None
    """
    with open("log.txt", 'a') as file:
        file.write(
            f"--- Run started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        file.write(f"Filename: {filename}.txt \n")
        file.write(f"Swept Gates: {[gate_out.label for gate_out in gates_out]} \n")
        file.write(f"Measured Input: {gate_in.label} \n")
        file.write(f"Start Voltage: {start_voltage} \n")
        file.write(f"End Voltage: {end_voltage} \n")
        file.write(f"Step Size: {step} \n")
        file.write("Initial Voltages of all outputs before sweep: \n")
        for output_gate in output_gates:
            file.write(
                f"{output_gate.name:>16} {output_gate.label:>16} {output_gate.voltage():>16} \n")
        file.write("\n")
        file.close()


# %% Animated plots


def sweep1D(gates_out: list[Gate], gate_in: Gate, start_voltage: Decimal, end_voltage: Decimal, step: Decimal,
            filename: str) -> None:
    """
    Perform a 1D voltage sweep and create an animated plot of the measurement.

    Args:
        gates_out (list[Gate]): List of output gates to sweep over.
        gate_in (Gate): Gate used to measure the input voltage.
        start_voltage (Decimal): Starting voltage value for the sweep.
        end_voltage (Decimal): Ending voltage value for the sweep.
        step (Decimal): Voltage increment for each step.
        filename (str): File name to save the results.

    Returns:
        None
    """
    if os.path.exists(f"{filename}.txt"):
        counter = 2
        while os.path.exists(f"{filename}_run{counter}.txt"):
            counter += 1
        filename = f"{filename}_run{counter}"

    x_label_gates = ""
    for gate_out in gates_out:
        x_label_gates += f"{gate_out.label} & "
    x_label_gates = x_label_gates[:-2]  # remove the trailing &

    # initialize voltage, wait for it to reach the start value
    for gate_out in gates_out:
        gate_out.voltage(start_voltage)

    while True:
        if all(abs(Decimal(gate_out.voltage()) - start_voltage) < Decimal(1e-8) for gate_out in gates_out):
            break
        time.sleep(0.1)
    time.sleep(2)

    # record the parameters
    log_params(gates_out, gate_in, start_voltage, end_voltage, step, filename)

    # initialize plotting
    fig, ax = plt.subplots()
    line, = ax.plot([], [])

    def init():
        line.set_data([], [])
        return line,

    # Perform the voltage sweep
    file = open(f"{filename}.txt", 'a+')
    file.write(f"{x_label_gates:>16} [V] {gate_in.label:>16} [uA] \n")

    voltages = []
    currents = []

    def update(frame):
        """
        Update function for the animation.

        Args:
            frame (int): Current frame number.

        Returns:
            line: Updated plot line with new data.
        """
        if start_voltage < end_voltage:
            # Increase voltage by one step every frame
            voltage = start_voltage + frame * step
            if voltage > end_voltage + Decimal(1e-8):
                return line
        else:
            voltage = start_voltage - frame * step
            if voltage < end_voltage - Decimal(1e-8):
                return line

        # Set voltage
        for gate_out in gates_out:
            gate_out.voltage(voltage)

        # Measure current
        current = gate_in.read_current(-1)  # -1 because of the inverting amplifier

        file.write(f"{voltage:>32}, {current:>32} \n")

        currents.append(current)
        voltages.append(voltage)

        line.set_data(voltages, currents)

        # update the scale of the plot
        ax.set_ylim(min(currents) - 0.01, max(currents) + 0.01)
        ax.set_xlim(min(voltages) - step, max(voltages) + step)

        ax.set_xlabel(f"{x_label_gates} [V]")
        ax.set_ylabel(f"{gate_in.label} [uA]")

        return line,

    num_frames = int(abs(end_voltage - start_voltage) / step) + 2

    # actually start the measurement
    global global_ani
    global_ani = FuncAnimation(fig, update, frames=np.arange(0, num_frames), init_func=init, blit=False, repeat=False)

    def on_close(_):
        """
        Close event handler to finalize the measurement and save the plot.

        Args:
            _: Unused argument for event handling.

        Returns:
            None
        """
        file.close()
        print('file closed')
        plt.savefig(f"{filename}.png")

    fig.canvas.mpl_connect('close_event', on_close)

    plt.show()  # keep the plot alive


# %% Set source
value_to_set = 0.01
t_s.voltage(value_to_set)

# %% Set reservoirs
value_to_set = 1.0
res_S_D.voltage(value_to_set)

# %% Measurements
# swept_terminal = [t_P1, t_P2, t_P3, t_barriers]
swept_terminal = [G8]
measured_input = t_D
filename = f'RT_{measured_input.label}_vs_{[item.label for item in swept_terminal]}'

start_voltage = Decimal(1)
end_voltage = Decimal(0)
step_size = Decimal(0.01)

sweep1D(swept_terminal, measured_input, start_voltage, end_voltage, step_size, filename)
