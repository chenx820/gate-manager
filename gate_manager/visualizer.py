import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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
        df = pd.read_csv(filename, sep='\s+', skiprows=1, header=None)
        if df.shape[1] != 3:
            raise ValueError(f"File format error! Expected 3 columns, but found {df.shape[1]}.")
        
        # Extract the data as floating point numbers
        self.y_values = df.iloc[:, 0].values.astype(float)
        self.x_values = df.iloc[:, 1].values.astype(float)
        self.currents = df.iloc[:, 2].values.astype(float)



    def viz2D(self, filename: str, x_min: float=None, x_max: float=None, y_min: float=None, y_max: float=None, 
              lower_threshold: float = 1, upper_threshold: float = 1):
        """
        Generates a 2D Coulomb Diamond plot from the given data file.
        """
        if filename:
            self.filename = filename
            self.read_2D_file(self.filename)
            self.currents *= 1000  # Convert currents to nA
        else:
            raise ValueError("Please provide a filename.")
        
        # Get unique V_G and V_SD values
        x_unique = np.unique(self.x_values)
        y_unique = np.unique(self.y_values)
        
        I_center_estimate = 0.1166 # Estimate for the center currents value from past experiments
        near_zero_indices = (self.currents > I_center_estimate - 0.001) & (self.currents < I_center_estimate + 0.001) # Consider values around 0V
        offset = np.mean(self.currents[near_zero_indices])
        currents_offset = self.currents - offset
        
        # Construct a 2D currents matrix (currents_grid)
        currents_grid = np.zeros((len(y_unique), len(x_unique)))

        # Fill the 2D grid with currents values
        for i in range(len(self.x_values)):
            x_idx = np.where(x_unique == self.x_values[i])[0][0]  # Find index for V_G
            y_idx = np.where(y_unique == self.y_values[i])[0][0]  # Find index for V_SD
            currents_grid[y_idx, x_idx] = currents_offset[i]  # Assign currents value to the correct grid position
        
        # Set range for plotting
        if x_min is None:
            x_min = x_unique.min()
        if x_max is None:
            x_max = x_unique.max()
        if y_min is None:
            y_min = y_unique.min()
        if y_max is None:
            y_max = y_unique.max()

        # Filter the valid range
        valid_x_idx = (x_unique >= x_min) & (x_unique <= x_max)
        x_unique = x_unique[valid_x_idx]
        currents_grid = currents_grid[:, valid_x_idx]  # Keep only selected range
        
        valid_y_idx = (y_unique >= y_min) & (y_unique <= y_max)
        y_unique = y_unique[valid_y_idx]
        currents_grid = currents_grid[valid_y_idx, :]  # Keep only selected range
        
        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(
            currents_grid,
            extent=[x_min, x_max, y_min, y_max],
            aspect='auto',
            origin='lower',
            cmap='bwr', 
            vmin=lower_threshold,
            vmax=upper_threshold
        )
        
        # Colorbar and labels
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        plt.colorbar(im, ax=ax, label=self.z_label)
        plt.savefig(self.filename.replace('.txt', '.png'), dpi=500)
    
    
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