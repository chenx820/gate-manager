import os
import numpy as np
import matplotlib.pyplot as plt
import pytest

from gate_manager.visualizer import Visualizer

# Sample file content with a header that splits into 6 tokens and will be combined into 3 labels.
SAMPLE_DATA = """t_bar_S1 [V] t_P1 [V] t_D [uA]
0.00000000 0.00000000 0.0001145153073594
0.00000000 0.05000000 0.0001172252697870
0.00000000 0.10000000 0.0001161052729003
0.05000000 0.00000000 0.0001136718783528
0.05000000 0.05000000 0.0001155693084002
0.05000000 0.10000000 0.0001160038053058
0.10000000 0.00000000 0.0001158740953542
0.10000000 0.05000000 0.0001144313835539
0.10000000 0.10000000 0.0001152709941380
"""

@pytest.fixture
def sample_file(tmp_path):
    """Create a temporary sample file containing 2D data."""
    file_path = tmp_path / "sample.txt"
    file_path.write_text(SAMPLE_DATA)
    return str(file_path)

def test_read_2D_file(sample_file):
    """Test that read_2D_file correctly parses the header and data."""
    vis = Visualizer()
    vis.read_2D_file(sample_file)
    # Expect the headers to be combined into:
    # y_label = "t_bar_S1 [V]", x_label = "t_P1 [V]", z_label = "t_D [uA]"
    assert vis.y_label == "t_bar_S1 [V]"
    assert vis.x_label == "t_P1 [V]"
    assert vis.z_label == "t_D [uA]"
    # Check that the data arrays have 9 entries.
    assert len(vis.x_values) == 9
    assert len(vis.y_values) == 9
    assert len(vis.currents) == 9

def test_viz2D(sample_file):
    """Test viz2D generates a PNG file for the given data."""
    vis = Visualizer()
    # Use arbitrary thresholds for testing.
    vis.viz2D(sample_file, lower_threshold=-1, upper_threshold=1)
    # The output file is saved as the filename with .txt replaced by .png.
    png_file = sample_file.replace(".txt", ".png")
    assert os.path.exists(png_file)
    # Clean up: remove the generated file and close figures.
    os.remove(png_file)
    plt.close('all')

def test_viz2D_slice_x_target(sample_file):
    """Test viz2D_slice using the x_target branch (i.e. slice along x)."""
    vis = Visualizer()
    # Call viz2D_slice with an x_target value; do not provide y_target.
    vis.viz2D_slice(filename=sample_file, x_target=0.05)
    # The method saves a file named <filename without .txt> + f'{target:.2f}.png'
    # Read the file to get x_values first.
    vis.read_2D_file(sample_file)
    # The second column (x_values) in our sample: 0.00000000, 0.05000000, 0.10000000, ...
    # The closest to 0.05 is 0.05.
    target = 0.05
    expected_png = sample_file.replace(".txt", "") + f"{target:.2f}.png"
    assert os.path.exists(expected_png)
    os.remove(expected_png)
    plt.close('all')

def test_viz2D_slice_invalid(sample_file):
    """Test that providing both x_target and y_target raises a ValueError."""
    vis = Visualizer()
    with pytest.raises(ValueError, match="Please choose only one target value."):
        vis.viz2D_slice(filename=sample_file, x_target=0.05, y_target=0.1)
