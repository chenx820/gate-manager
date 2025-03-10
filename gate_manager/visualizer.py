# -*- coding: utf-8 -*-
"""
Visualizer class for plotting 2D Coulomb Diamond plots and 1D slices.

This class reads a 2D data file and generates a 2D Coulomb Diamond plot.
It also allows the user to plot 1D slices of the data at a specific V_SD or V_G value.
             
Created on Wed Mar 03 15:04:20 2025
@author:
Chen Huang <chen.huang23@imperial.ac.uk>
John Michniewicz <j.michniewicz23@imperial.ac.uk>
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


class Visualizer:
    def __init__(self):
        self.x_values = []
        self.y_values = []
        self.currents = []
        
        self.x_label = None
        self.y_label = None
        self.z_label = r'$I$ [nA]'
        self.filename = None

    def read_2D_file(self, filename: str):
        # Read the first line to get header information
        with open(filename, 'r') as f:
            header_line = f.readline().strip()
        
        # Split the header by whitespace
        header_tokens = header_line.split()
        # If there are an even number of tokens and more than 3 tokens, assume each header is made of two tokens.
        if len(header_tokens) % 2 == 0 and len(header_tokens) > 3:
            header = []
            for i in range(0, len(header_tokens), 2):
                header.append(header_tokens[i] + " " + header_tokens[i+1])
        else:
            header = header_tokens
        
        if len(header) != 3:
            raise ValueError(f"Header format error! Expected 3 labels, but found {len(header)}: {header}")
        
        # Assign the headers to y_label, x_label, and z_label respectively
        self.y_label, self.x_label, self.z_label = header
    
        # Read the rest of the file into a DataFrame
        self.df = pd.read_csv(filename, sep=r'\s+', skiprows=1, names=["Y", "X", "Z"])  # Assign column names
        
        z_offset = 0.1166 # Estimate for the center currents value from past experiments
        
        # Construct a 2D currents matrix (currents_grid)
        self.df_pivot = self.df.pivot(index="Y", columns="X", values="Z")
        
        # Convert Pandas DataFrame to NumPy arrays
        self.x_values = self.df_pivot.columns.values  # X-axis values
        self.y_values = self.df_pivot.index.values  # Y-axis values
        self.z_matrix = self.df_pivot.values  # Z values in 2D array format


    def viz2D(self, filename: str, z_min: float=None, z_max: float=None):
        """
        Generates a 2D Coulomb Diamond plot from the given data file.
        """
        if filename:
            self.filename = filename
            self.read_2D_file(self.filename)
        else:
            raise ValueError("Please provide a filename.")
        
        # Z-axis settings
        if z_min is None:
            z_min = self.z_matrix.min()
        if z_max is None:
            z_max = self.z_matrix.max()
        z_level = 500     # Number of levels in Z-axis

        # Define custom colormap
        colorsbar = ['#02507d', '#ede8e5', '#b5283b']
        cm = LinearSegmentedColormap.from_list('', colorsbar, N=z_level)


        # Plotting
        fig, ax = plt.subplots(figsize=(12, 7))
        img = ax.imshow(
            self.z_matrix, vmin=z_min, vmax=z_max, 
            cmap=cm, aspect='auto', origin='lower',    
            extent=[self.x_values[0], self.x_values[-1], self.y_values[0], self.y_values[-1]],
            interpolation='none',
        )
        
        # Plot decorators
        plt.style.use('fivethirtyeight')
        plt.rc('legend', fontsize=22, framealpha = 0.9)
        plt.rc('xtick', labelsize=24, color='#2C3E50') 
        plt.rc('ytick', labelsize=24, color='#2C3E50')
        
        fig.patch.set_facecolor('white')
        
        # Colorbar customization
        barticks = np.linspace(z_min, z_max, 5)  # Generate bar ticks
        barticks = np.around(barticks, 2)        # Round to 2 decimal places
        barticks_labels = [str(label) for label in barticks]
        barticks_labels[0] = f"< {barticks[0]}"
        barticks_labels[-1] = f"> {barticks[-1]}"
        
        cbar = fig.colorbar(img, ticks=barticks, pad=0.005, extend='both')
        cbar.ax.set_yticklabels(barticks_labels)  # Custom tick labels
        cbar.ax.set_title(f'         {self.z_label}', fontsize=28, pad=10)  # Colorbar title
        cbar.ax.tick_params(direction='in', width=2, length=5, labelsize=22)  # Colorbar ticks
        
        # Border
        ax.spines['right'].set_color('#2C3E50')
        ax.spines['bottom'].set_color('#2C3E50')
        ax.spines['left'].set_color('#2C3E50')
        ax.spines['top'].set_color('#2C3E50')
        
        # Axes labels
        ax.set_xlabel(self.x_label, color='#2C3E50', fontsize=32) 
        ax.set_ylabel(self.y_label, color='#2C3E50', fontsize=32)
        
        #Ticks
        ax.tick_params(axis='y', direction='in', width=4, length=10 , pad=10 , right=True)
        ax.tick_params(axis='x', direction='in', width=4, length=10 , pad=10 , top=False)

        plt.tight_layout()
        fig.savefig(self.filename.replace('.txt', '.png'), dpi=300, bbox_inches='tight')
        
    
    def viz2D_slice(self, filename: str=None, x_target: float=None, y_target: float=None):
        """
        Plots 1D currents vs. V_G at a specific V_SD value.
        """
        if filename:
            self.filename = filename
            self.read_2D_file(self.filename)
            self.currents *= 1000  # Convert currents to nA
        else:
            raise ValueError("Please provide a filename.")

        if x_target and y_target:
            raise ValueError("Please choose only one target value.")
        
        # Extract data for the chosen y_target
        elif y_target:
            # Find the closest index to the target
            idx = np.abs(self.y_values - self.y_target).argmin()
            target = self.y_values[idx]
            x_selected = self.x_values[self.y_values == self.y_values[idx]]
            currents_selected = self.currents[self.y_values == self.y_values[idx]]

            # Sort values in case the order is mixed
            sorted_indices = np.argsort(x_selected)
            voltages_selected = x_selected[sorted_indices]
            label_selected = self.x_label
            currents_selected = currents_selected[sorted_indices]
            
        # Extract data for the chosen x_target
        elif x_target:
            idx = np.abs(self.x_values - x_target).argmin()
            target = self.x_values[idx]
            y_selected = self.y_values[self.x_values == self.x_values[idx]]
            currents_selected = self.currents[self.x_values == self.x_values[idx]]

            # Sort values in case the order is mixed
            sorted_indices = np.argsort(y_selected)
            voltages_selected = y_selected[sorted_indices]
            label_selected = self.y_label
            currents_selected = currents_selected[sorted_indices]
            
        else:
            raise ValueError("Please choose a target value.")

        # Plotting
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(voltages_selected, currents_selected, linestyle='-', color='b')
        ax.set_xlabel(label_selected, fontsize=14)
        ax.set_ylabel(self.z_label, fontsize=14)
        plt.grid()
        plt.savefig(self.filename.replace('.txt', '')+f'{target:.2f}.png', dpi=300)
        plt.show()