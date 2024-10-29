# -*- coding: utf-8 -*-
"""
This script performs voltage sweep experiments using a Nanonis instance.

"""

from nanonis_spm import Nanonis, Gate, GatesGroup
import socket
import time
import matplotlib.pyplot as plt
import os
from datetime import datetime
from decimal import Decimal
from tqdm import tqdm

# Create a socket connection to Nanonis
connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
connection.connect(("192.168.236.1", 6501))

# Create Nanonis instance for controlling the device
nanonisInstance = Nanonis(connection)

last_update_time = time.time()

# %% Define parameters for the experiment
slew_rate = 0.1
device = "Semiqon 36"
amplifier = (-1) * 10 ** 6  # amplification factor applied to the measured current, unit [uA]

# %% Define gates
# Define output gates for controlling voltage
# These gates represent specific terminals to apply or read voltages from.

# Define output gates
t_P1 = Gate(name='output1', label='t_P1', read_index=24, write_index=1, nanonisInstance=nanonisInstance)
t_P2 = Gate(name='output2', label='t_P2', read_index=25, write_index=2, nanonisInstance=nanonisInstance)
t_P3 = Gate(name='output3', label='t_P3', read_index=26, write_index=3, nanonisInstance=nanonisInstance)
t_P4 = Gate(name='output4', label='t_P4', read_index=27, write_index=4, nanonisInstance=nanonisInstance)
t_barriers = Gate(name='output5', label='t_barriers', read_index=28, write_index=5, nanonisInstance=nanonisInstance)
res_S_D = Gate(name='output6', label='res_S_D', read_index=29, write_index=6, nanonisInstance=nanonisInstance)
t_s = Gate(name='output7', label='t_s', read_index=30, write_index=7, nanonisInstance=nanonisInstance)
G8 = Gate(name='output8', label='G8', read_index=31, write_index=8, nanonisInstance=nanonisInstance)

# Grouping gates for easier voltage control
output_gates = GatesGroup([t_P1, t_P2, t_P3, t_P4, t_barriers, res_S_D, t_s, G8])
finger_gates = GatesGroup([t_P1, t_P2, t_P3, t_P4, t_barriers])
plunger_gates = GatesGroup([t_P1, t_P2, t_P3, t_P4])

# Define input gates for reading current measurements
t_D = Gate(name='input1', label='t_D', read_index=0, nanonisInstance=nanonisInstance)
b_D = Gate(name='input2', label='b_D', read_index=1, nanonisInstance=nanonisInstance)
SD3 = Gate(name='input3', label='SD3', read_index=2, nanonisInstance=nanonisInstance)
SD4 = Gate(name='input4', label='SD4', read_index=3, nanonisInstance=nanonisInstance)
SD5 = Gate(name='input5', label='SD5', read_index=4, nanonisInstance=nanonisInstance)
SD6 = Gate(name='input6', label='SD6', read_index=5, nanonisInstance=nanonisInstance)
SD7 = Gate(name='input7', label='SD7', read_index=6, nanonisInstance=nanonisInstance)
SD8 = Gate(name='input8', label='SD8', read_index=7, nanonisInstance=nanonisInstance)

input_gates = GatesGroup([t_D, b_D, SD3, SD4, SD5, SD6, SD7, SD8])


# %% Log file
def log_params(gates_out: GatesGroup, gate_in: Gate, start_voltage: Decimal, end_voltage: Decimal, step: Decimal,
               filename: str) -> None:
    """
    Log the parameters of the sweep to a file.

    Args:
        gates_out (GatesGroup): List of output gates that are being swept.
        gate_in (Gate): The input gate whose current is measured.
        start_voltage (Decimal): Starting voltage of the sweep.
        end_voltage (Decimal): End voltage of the sweep.
        step (Decimal): Step size for the sweep.
        filename (str): Name of the file where results are saved.
    """
    with open("log.txt", 'a') as file:
        file.write(
            f"--- Run started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        file.write(f"Filename: {filename}.txt \n")
        file.write(f"Device: {device} \n")
        file.write(f"Amplifier: {amplifier} \n")
        file.write(f"Swept Gates: {[gate_out.label for gate_out in gates_out.gates]} \n")
        file.write(f"Measured Input: {gate_in.label} \n")
        file.write(f"Start Voltage: {float(start_voltage)} [V] \n")
        file.write(f"End Voltage: {float(end_voltage)} [V] \n")
        file.write(f"Step Size: {float(step)} [V] \n")
        file.write(f"Slew Rate: {slew_rate} [V/s] \n")
        file.write("Initial Voltages of all outputs before sweep: \n")
        for output_gate in output_gates.gates:
            file.write(
                f"{output_gate.name:>16} {output_gate.label:>16} {output_gate.voltage():>16.8} [V] \n")
        file.write("\n")


# %% Animated plots

def sweep1D(swept_terminal: GatesGroup, measured_input: Gate, start_voltage: float, end_voltage: float, step: float,
            temperature: str, initial_state: dict) -> None:
    """
    Perform a 1D voltage sweep and create an animated plot of the measurement.

    Args:
        swept_terminal (GatesGroup): List of output gates to sweep over.
        measured_input (Gate): Gate used to measure the input voltage.
        start_voltage (float): Starting voltage value for the sweep.
        end_voltage (float): Ending voltage value for the sweep.
        step (float): Voltage increment for each step.
        temperature (str): Experimental temperature setting used for filename.
        initial_state (dict): Initial voltage state for each gate in the setup.
    """
    # Ensure voltage in safty range (-2, 2)
    min_voltage = -2
    max_voltage = 2
    if start_voltage < min_voltage or start_voltage > max_voltage:
        raise ValueError(f"Start voltage is out of range {(min_voltage, max_voltage)}.")
    if end_voltage < min_voltage or end_voltage > max_voltage:
        raise ValueError(f"End voltage is out of range {(min_voltage, max_voltage)}.")
    for gate, voltage in initial_state.items():
        if voltage < min_voltage or voltage > max_voltage:
            raise ValueError(f"{gate} initial voltage {voltage} is out of range {(min_voltage, max_voltage)}.")

    # Define the file name
    filename = f"{temperature}_{measured_input.label}_vs_{[gate.label for gate in swept_terminal.gates]}"
    if os.path.exists(f"{filename}.txt"):
        counter = 2
        while os.path.exists(f"{filename}_run{counter}.txt"):
            counter += 1
        filename = f"{filename}_run{counter}"

    # Initializing plot
    x_label = " & ".join(gate.label for gate in swept_terminal.gates)
    start_voltage = Decimal(start_voltage)
    end_voltage = Decimal(end_voltage)
    step = Decimal(step)

    # Set initial voltages
    pbar = tqdm(total=len(output_gates.gates) + len(initial_state.items()) + len(swept_terminal.gates),
                desc="[INFO] Ramping voltage", ncols=80, bar_format="{desc}: [{bar}] {percentage:3.0f}%",
                ascii="▪▫")  # progress bar
    output_gates.turn_off()
    pbar.update(len(output_gates.gates))
    preset = [(gate, initial_voltage) for gate_label, initial_voltage in initial_state.items() for gate in
              output_gates.gates if gate.label == gate_label]
    for gate, initial_voltage in preset:
        gate.voltage(initial_voltage, False)

    # Wait for initial voltages to stabilize
    while not all([gate.is_at_target_voltage(voltage) for gate, voltage in preset]):
        time.sleep(0.1)
    pbar.update(len(initial_state.items()))
    # Initialize sweep parameters and plotting
    swept_terminal.voltage(start_voltage)
    pbar.update(len(swept_terminal.gates))
    pbar.close()
    time.sleep(1)

    fig, ax = plt.subplots()
    line, = ax.plot([], [])
    ax.set_xlabel(f"{x_label} [V]")
    ax.set_ylabel(f"{measured_input.label} [uA]")
    voltages, currents = [], []
    voltage = start_voltage

    with open(f"{filename}.txt", 'a') as file:
        file.write(f"{x_label:>20} [V] {measured_input.label:>19} [uA] \n")

    # record the parameters
    log_params(swept_terminal, measured_input, start_voltage, end_voltage, step, filename)

    # actually start the measurement
    print(
        f"[INFO] Start sweeping {[gate.label for gate in swept_terminal.gates]} from {float(start_voltage)} [V] to {float(end_voltage)} [V]. ")

    # Execute sweep and record data
    total = int(abs(end_voltage - start_voltage) / step + 1)
    pbar = tqdm(total=total, desc="[INFO] Sweeping", ncols=80, bar_format="{desc}: [{bar}] {percentage:3.0f}%",
                ascii="▪▫")  # progress bar
    frame = 0
    while True:
        swept_terminal.voltage(voltage)
        voltages.append(voltage)
        current = measured_input.read_current(-1)  # -1 because of the inverting amplifier
        currents.append(current)

        with open(f"{filename}.txt", 'a') as file:
            file.write(f"{round(voltage, 16):>24} {round(current, 16):>24} \n")
        ax.set_xlim(min(voltages) - step, max(voltages) + step)
        ax.set_ylim(min(currents) - Decimal(0.01), max(currents) + Decimal(0.01))
        line.set_data(voltages, currents)
        plt.draw()
        plt.pause(0.1)
        frame += 1
        pbar.update(1)

        if (start_voltage < end_voltage and voltage > end_voltage - Decimal(1e-6)) or (
                start_voltage > end_voltage and voltage < end_voltage + Decimal(1e-6)):
            pbar.close()
            break
        voltage = start_voltage + frame * step if start_voltage < end_voltage else start_voltage - frame * step

    plt.savefig(f"{filename}.png", dpi=300)
    print("[INFO] Data collection complete and figure saved. \n")


# %% Parameters
uniform = {
    'swept_terminal': finger_gates,
    'measured_input': t_D,
    'start_voltage': -1,
    'end_voltage': 1,
    'step': 0.1,
    'temperature': 'RT',
    'initial_state': {
        't_s': 0.01,  # source voltage
        'res_S_D': 1.0,  # reservoir voltage
    }
}

pinch_off = {
    'swept_terminal': GatesGroup([t_P1]),
    'measured_input': t_D,
    'start_voltage': 1.0,
    'end_voltage': -1.0,
    'step': 0.1,
    'temperature': 'RT',
    'initial_state': {
        't_s': 0.01,
        'res_S_D': 1.0,
        't_P1': 1.0,
        't_P2': 1.0,
        't_P3': 1.0,
        't_P4': 1.0,
        't_barriers': 1.0,
    }
}

# %% Run
sweep1D(**uniform)

for gate in finger_gates.gates:
    pinch_off['swept_terminal'] = GatesGroup([gate])
    sweep1D(**pinch_off)

# %% Turn off
output_gates.turn_off()
