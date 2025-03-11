import pytest
from gate_manager.connection import (
    SemiqonLine,
    SemiqonLinesConnection,
    NanonisSource,
    NanonisSourceConnection
)

# --- Tests for SemiqonLine class ---

def test_semiqon_line_default_label():
    """
    Test that a SemiqonLine instance has a default label of None.
    """
    line = SemiqonLine()
    assert line.label is None

def test_semiqon_line_custom_label():
    """
    Test that a SemiqonLine instance correctly sets a custom label.
    """
    custom_label = "Test Label"
    line = SemiqonLine(label=custom_label)
    assert line.label == custom_label

# --- Tests for SemiqonLinesConnection class ---

def test_semiqon_lines_connection_length_and_labels():
    """
    Test that SemiqonLinesConnection correctly initializes all SemiqonLine instances and their labels.
    """
    connection = SemiqonLinesConnection()
    # According to the source code, the lines list should contain 1 (empty) + 12 (top) + 12 (bottom) = 25 elements.
    assert len(connection.lines) == 25

    # The first element should be empty (None label)
    assert connection.lines[0].label is None

    expected_top_labels = [
        't_D', 't_bar_4D', 't_P4', 't_bar_34', 't_P3', 't_bar_23',
        't_P2', 't_bar_12', 't_P1', 't_bar_S1', 't_s', 'res_S'
    ]
    expected_bottom_labels = [
        'b_S', 'b_bar_S1', 'b_P1', 'b_bar_12', 'b_P2', 'b_bar_23',
        'b_P3', 'b_bar_34', 'b_P4', 'b_bar_4D', 'b_D', 'res_D'
    ]

    # Check top line labels (indices 1 to 12)
    for i, label in enumerate(expected_top_labels, start=1):
        assert connection.lines[i].label == label

    # Check bottom line labels (indices 13 to 24)
    for i, label in enumerate(expected_bottom_labels, start=13):
        assert connection.lines[i].label == label

# --- Tests for NanonisSource class ---

def test_nanonis_source_properties():
    """
    Test that NanonisSource properties are correctly assigned.
    """
    test_label = "Test Source"
    test_read_index = 5
    test_write_index = 10
    dummy_nanonis = object()  # Dummy object to simulate a Nanonis instance

    source = NanonisSource(
        label=test_label,
        read_index=test_read_index,
        write_index=test_write_index,
        nanonisInstance=dummy_nanonis
    )
    assert source.label == test_label
    assert source.read_index == test_read_index
    assert source.write_index == test_write_index
    assert source.nanonisInstance == dummy_nanonis

# --- Tests for NanonisSourceConnection class ---

def test_nanonis_source_connection_outputs():
    """
    Test that the outputs list in NanonisSourceConnection is initialized correctly.
    """
    conn = NanonisSourceConnection(nanonisInstance=None)
    # The outputs list should contain 9 elements: 1 empty source + 8 output sources.
    assert len(conn.outputs) == 9

    # The first output should be empty.
    assert conn.outputs[0].label is None

    # Verify each output source's properties.
    for i in range(1, 9):
        source = conn.outputs[i]
        expected_label = f'Nanonis output{i}'
        expected_read_index = 23 + i  # For example, output1 should have read_index 24.
        expected_write_index = i
        assert source.label == expected_label
        assert source.read_index == expected_read_index
        assert source.write_index == expected_write_index
        assert source.nanonisInstance is None

def test_nanonis_source_connection_inputs():
    """
    Test that the inputs list in NanonisSourceConnection is initialized correctly.
    """
    conn = NanonisSourceConnection(nanonisInstance=None)
    # The inputs list should contain 9 elements: 1 empty source + 8 input sources.
    assert len(conn.inputs) == 9

    # The first input should be empty.
    assert conn.inputs[0].label is None

    # Verify each input source's properties.
    for i in range(1, 9):
        source = conn.inputs[i]
        expected_label = f'Nanonis input{i}'
        expected_read_index = i - 1  # For example, input1 should have read_index 0.
        assert source.label == expected_label
        assert source.read_index == expected_read_index
        # Input sources should not have write_index (it should be None)
        assert source.write_index is None
        assert source.nanonisInstance is None
