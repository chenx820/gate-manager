# tests/test_connection.py

import pytest
from unittest.mock import MagicMock

# Adjust the import path if necessary.
from gate_manager.connection import (
    SemiqonLine,
    SemiqonLinesConnection,
    NanonisSource,
    NanonisSourceConnection,
)

#############################
#    SemiqonLine Tests      #
#############################

def test_semiqon_line_with_label():
    """Test that a SemiqonLine constructed with a label sets the attribute correctly."""
    line = SemiqonLine(label="TestLabel")
    assert line.label == "TestLabel"

def test_semiqon_line_default():
    """Test that a SemiqonLine without a label has label set to None."""
    line = SemiqonLine()
    assert line.label is None

#############################
# SemiqonLinesConnection Tests
#############################

def test_semiqon_lines_connection_lines():
    """
    Test that SemiqonLinesConnection creates the expected number of lines 
    and that certain labels are set correctly.
    """
    connection = SemiqonLinesConnection()
    # There are 1 empty line + 12 top lines + 12 bottom lines = 25 total lines.
    total_expected_lines = 25
    assert len(connection.lines) == total_expected_lines

    # First line should be empty (label is None)
    assert connection.lines[0].label is None

    # Check top lines (indices 1 to 12)
    expected_top_labels = [
        't_D',
        't_bar_4D',
        't_P4',
        't_bar_34',
        't_P3',
        't_bar_23',
        't_P2',
        't_bar_12',
        't_P1',
        't_bar_S1',
        't_s',
        'res_S',
    ]
    for i, expected_label in enumerate(expected_top_labels, start=1):
        assert connection.lines[i].label == expected_label

    # Check bottom lines (indices 13 to 24)
    expected_bottom_labels = [
        'b_S',
        'b_bar_S1',
        'b_P1',
        'b_bar_12',
        'b_P2',
        'b_bar_23',
        'b_P3',
        'b_bar_34',
        'b_P4',
        'b_bar_4D',
        'b_D',
        'res_D',
    ]
    for i, expected_label in enumerate(expected_bottom_labels, start=13):
        assert connection.lines[i].label == expected_label

#############################
#   NanonisSource Tests     #
#############################

def test_nanonis_source_attributes():
    """Test that NanonisSource correctly sets its attributes."""
    dummy_instance = MagicMock()
    source = NanonisSource(label="Source1", read_index=5, write_index=2, nanonisInstance=dummy_instance)
    assert source.label == "Source1"
    assert source.read_index == 5
    assert source.write_index == 2
    assert source.nanonisInstance is dummy_instance

#############################
# NanonisSourceConnection Tests
#############################

@pytest.fixture
def dummy_nanonis_instance():
    """Provide a dummy nanonis instance using a MagicMock."""
    return MagicMock()

def test_nanonis_source_connection_outputs(dummy_nanonis_instance):
    """
    Test that NanonisSourceConnection creates the expected outputs.
    There should be 9 outputs with the proper labels and indices.
    """
    connection = NanonisSourceConnection(nanonisInstance=dummy_nanonis_instance)
    # Check that there are 9 output sources.
    assert len(connection.outputs) == 9

    # First output is empty
    assert connection.outputs[0].label is None

    # Check the details of the first non-empty output
    output1 = connection.outputs[1]
    assert output1.label == 'Nanonis output1'
    assert output1.read_index == 24
    assert output1.write_index == 1
    assert output1.nanonisInstance is dummy_nanonis_instance

def test_nanonis_source_connection_inputs(dummy_nanonis_instance):
    """
    Test that NanonisSourceConnection creates the expected inputs.
    There should be 9 inputs with the proper labels and read indices.
    """
    connection = NanonisSourceConnection(nanonisInstance=dummy_nanonis_instance)
    # Check that there are 9 input sources.
    assert len(connection.inputs) == 9

    # First input is empty
    assert connection.inputs[0].label is None

    # Check the details of the first non-empty input
    input1 = connection.inputs[1]
    assert input1.label == 'Nanonis input1'
    assert input1.read_index == 0
    # write_index is not provided for inputs in the constructor, so it should be None.
    assert input1.write_index is None
    assert input1.nanonisInstance is dummy_nanonis_instance
