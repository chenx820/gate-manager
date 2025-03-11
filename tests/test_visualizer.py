import os
import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend for testing
import matplotlib.pyplot as plt

from gate_manager.visualizer import Visualizer

# ---------------------------
# Fixtures for Test Data Files
# ---------------------------

@pytest.fixture
def valid_2d_file(tmp_path):
    """
    Create a temporary valid 2D data file.
    
    The file has a header with three labels and four data rows:
        Header: "Y_label X_label Z_label"
        Data:
            0 0 1
            0 1 2
            1 0 3
            1 1 4
    """
    file_content = (
        "Y_label X_label Z_label\n"
        "0 0 1\n"
        "0 1 2\n"
        "1 0 3\n"
        "1 1 4\n"
    )
    file_path = tmp_path / "test_data.txt"
    file_path.write_text(file_content)
    return file_path

@pytest.fixture
def invalid_header_file(tmp_path):
    """
    Create a temporary file with an invalid header (only 2 tokens).
    """
    file_content = (
        "Y_label X_label\n"
        "0 0 1\n"
        "0 1 2\n"
    )
    file_path = tmp_path / "invalid_header.txt"
    file_path.write_text(file_content)
    return file_path

# ---------------------------
# Tests for Visualizer.read_2D_file()
# ---------------------------

def test_read_2D_file_valid(valid_2d_file):
    """
    Test that read_2D_file correctly reads a valid 2D data file.
    """
    viz = Visualizer()
    viz.read_2D_file(str(valid_2d_file))
    
    # Verify that the header tokens are assigned correctly.
    # Note: self.y_label gets the first token, self.x_label the second, self.z_label the third.
    assert viz.y_label == "Y_label"
    assert viz.x_label == "X_label"
    assert viz.z_label == "Z_label"
    
    # Check that the pivoted data has the correct shape and values.
    # Expect x_values: [0, 1], y_values: [0, 1] and z_matrix as [[1, 2], [3, 4]].
    np.testing.assert_allclose(viz.x_values, [0, 1], atol=1e-6)
    np.testing.assert_allclose(viz.y_values, [0, 1], atol=1e-6)
    expected_matrix = np.array([[1, 2],
                                [3, 4]])
    np.testing.assert_allclose(viz.z_matrix, expected_matrix, atol=1e-6)

def test_read_2D_file_invalid_header(invalid_header_file):
    """
    Test that read_2D_file raises a ValueError when the header does not contain 3 labels.
    """
    viz = Visualizer()
    with pytest.raises(ValueError):
        viz.read_2D_file(str(invalid_header_file))

# ---------------------------
# Tests for Visualizer.viz2D()
# ---------------------------

def test_viz2D_creates_png(tmp_path, valid_2d_file, monkeypatch):
    """
    Test that viz2D creates a PNG file from the 2D data file.
    
    The test redirects the working directory to a temporary path so that the output PNG
    is created in that temporary directory.
    """
    # Redirect os.getcwd() to the temporary path.
    monkeypatch.setattr(os, "getcwd", lambda: str(tmp_path))
    
    viz = Visualizer()
    # Call viz2D with the valid file.
    viz.viz2D(str(valid_2d_file), z_min=0, z_max=5)
    
    # The output filename is based on the input file name with '.txt' replaced by '.png'
    output_png = str(valid_2d_file).replace('.txt', '.png')
    assert os.path.isfile(output_png)
    
    # Clean up the generated file.
    os.remove(output_png)
    plt.close('all')

def test_viz2D_no_filename():
    """
    Test that viz2D raises a ValueError when no filename is provided.
    """
    viz = Visualizer()
    with pytest.raises(ValueError):
        viz.viz2D("")
