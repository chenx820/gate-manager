# -*- coding: utf-8 -*-
"""
This script performs voltage sweep experiments using a Nanonis instance.
"""
from nanonis_tramea import Nanonis
from gate_manager.gate import Gate, GatesGroup
from gate_manager.connection import NanonisSourceConnection, SemiqonLinesConnection
from gate_manager.sweeper import Sweeper
import socket
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

# Create a socket connection to Nanonis
connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
connection.connect(("192.168.236.1", 6501))

# Create Nanonis instance for controlling the device
nanonisInstance = Nanonis(connection)

# Connection
nanonis_o = NanonisSourceConnection(nanonisInstance).outputs
nanonis_i = NanonisSourceConnection(nanonisInstance).inputs
lines = SemiqonLinesConnection().lines

# %% Define gates

# top channel 
t_P1 = Gate(source=nanonis_o[1], lines=[lines[9]])  # tP1: 9
t_bar_S1 = Gate(source=nanonis_o[2], lines=[lines[10]])  #t_bar_S1
t_bar_12 = Gate(source=nanonis_o[3], lines=[lines[8]])  #t_bar_1-2
t_global =  Gate(source=nanonis_o[5], lines=[lines[2], lines[3], lines[4], lines[5], lines[6], lines[7]])

# sources and reservoirs
t_b_s = Gate(source=nanonis_o[7], lines=[lines[11], lines[13]])  # bs,ts: 11,13
res_S_D = Gate(source=nanonis_o[8], lines=[lines[12], lines[24]])  # resS,resD: 12,24

# Grouping gates for easier voltage control
outputs = GatesGroup([t_P1, t_bar_S1, t_bar_12, t_global, t_b_s, res_S_D])
fingers = GatesGroup([t_P1, t_bar_S1, t_bar_12, t_global])

# %% Define input gates for reading currents measurements

t_D = Gate(source=nanonis_i[1], lines=[lines[1]], amplification=-1e8)
b_D = Gate(source=nanonis_i[2], lines=[lines[23]], amplification=-1e7)
SD3 = Gate(source=nanonis_i[3])
SD4 = Gate(source=nanonis_i[4])
SD5 = Gate(source=nanonis_i[5])
SD6 = Gate(source=nanonis_i[6])
SD7 = Gate(source=nanonis_i[7])
SD8 = Gate(source=nanonis_i[8])

# this should be automatic for the input gates grouping. 
inputs = GatesGroup([t_D, b_D, SD4, SD5, SD6, SD7, SD8])


# %% Define parameters for the experiment
slew_rate = 0.1 
for i in range(8):
    nanonisInstance.UserOut_SlewRateSet(i+1, slew_rate)
params = {
    'device': "Semiqon 36",
    'temperature': "CT"
    }
sweeper = Sweeper(outputs, inputs, **params)


# %% 1D sweep

params_1d = {
    'swept_outputs': GatesGroup([t_P1]),
    'measured_inputs': GatesGroup([t_D]),
    'start_voltage': [-0.6, 'V'],
    'end_voltage': [0, 'V'],
    'step': [0.5, 'mV'],
    'initial_state': [
        [t_b_s, 0.4, 'mV'],
        [res_S_D, 2, 'V'],
        [t_bar_S1, 1.0, 'V'],
        [t_bar_12, 1.0, 'V'],
        [t_global, 0.8, 'V']
        ],
    'current_unit': 'nA',
    'comments': 'test',
    'is_show': True
    }
    
sweeper.sweep1D(**params_1d)


# %% 2D sweep

params_2d = {
        'X_swept_outputs': GatesGroup([t_P1]),
        'X_start_voltage': [-0.3, 'V'],
        'X_end_voltage': [0.3, 'V'],
        'X_step': [2, 'mV'],
        'Y_swept_outputs': GatesGroup([res_S_D]),
        'Y_start_voltage': [0.5, 'V'],
        'Y_end_voltage': [1.2, 'V'],
        'Y_step': [10, 'mV'],
        'measured_inputs': GatesGroup([t_D]),
        'initial_state': [
            [t_b_s, 0.4, 'mV'], 
            [t_bar_S1, 1, 'V'],
            [t_bar_12, 1, 'V'],
            [t_global, 0.8, 'V']
        ],
        'current_unit': 'nA',
        'comments': 'diamond',
        'is_show': True
        }
sweeper.sweep2D(**params_2d)


# %% Turn off
sweeper.cleanup()