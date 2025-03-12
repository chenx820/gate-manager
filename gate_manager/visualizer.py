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
        self.z_label = None
        self.filename = None

    def read_2D_file(self, filename: str):
        # Read the first line to get header information
        with open(f"data/{filename}", 'r') as f:
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
        self.df = pd.read_csv(f"data/{filename}", sep=r'\s+', skiprows=1, names=["Y", "X", "Z"])  # Assign column names
        
        z_offset = 0.1166 # Estimate for the center currents value from past experiments
        
        # Construct a 2D currents matrix (currents_grid)
        self.df_pivot = self.df.pivot(index="Y", columns="X", values="Z")
        
        # Convert Pandas DataFrame to NumPy arrays
        self.x_values = self.df_pivot.columns.values  # X-axis values
        self.y_values = self.df_pivot.index.values  # Y-axis values
        self.z_matrix = self.df_pivot.values  # Z values in 2D array format


    def viz2D(self, filename: str, z_min: float=None, z_max: float=None, is_show: bool=True):
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
        plt.rc('legend', fontsize=10, framealpha = 0.9)
        plt.rc('xtick', labelsize=12, color='#2C3E50') 
        plt.rc('ytick', labelsize=12, color='#2C3E50')
        
        fig.patch.set_facecolor('white')
        
        # Colorbar customization
        barticks = np.linspace(z_min, z_max, 5)  # Generate bar ticks
        barticks = np.around(barticks, 4)        # Round to 4 decimal places
        barticks_labels = [str(label) for label in barticks]
        barticks_labels[0] = f"< {barticks[0]}"
        barticks_labels[-1] = f"> {barticks[-1]}"
        
        cbar = fig.colorbar(img, pad=0.005, extend='both')
        cbar.set_ticks(barticks)  # Custom tick marks
        cbar.ax.set_yticklabels(barticks)   # Custom tick labels
        cbar.ax.set_title(f'         {self.z_label}', fontsize=14, pad=10)  # Colorbar title
        cbar.ax.tick_params(direction='in', width=2, length=5, labelsize=10)  # Colorbar ticks
        
        # Border
        ax.spines['right'].set_color('#2C3E50')
        ax.spines['bottom'].set_color('#2C3E50')
        ax.spines['left'].set_color('#2C3E50')
        ax.spines['top'].set_color('#2C3E50')
        
        # Axes labels
        ax.set_xlabel(self.x_label, color='#2C3E50', fontsize=14) 
        ax.set_ylabel(self.y_label, color='#2C3E50', fontsize=14)
        
        #Ticks
        ax.tick_params(axis='y', direction='in', width=4, length=10 , pad=10 , right=True, labelsize=14)
        ax.tick_params(axis='x', direction='in', width=4, length=10 , pad=10 , top=False, labelsize=14)

        plt.tight_layout()
        plt.savefig("figures/"+self.filename.replace('.txt', '.png'), dpi=300, bbox_inches='tight')
        print("[INFO]: 2D plot saved.")
        if is_show:
            plt.show()
        
        
        
    