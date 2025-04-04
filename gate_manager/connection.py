# -*- coding: utf-8 -*-
"""
Connection Classes for Semiqon and Nanonis System.

This module defines classes for setting up and managing connections with the Nanonis system, including
`SemiqonLine`, `SemiqonLinesConnection`, `NanonisSource`, and `NanonisSourceConnection`. These classes
enable structured voltage control across various terminals within the Nanonis system.

Classes:
    SemiqonLine: Represents a single line within the Semiqon system with an optional label.
    SemiqonLinesConnection: Manages a collection of Semiqon lines, organizing them as top or bottom lines.
    NanonisSource: Defines an input/output source for the Nanonis system.
    NanonisSourceConnection: Organizes input and output sources for the Nanonis system.

Created on Tue Oct 22 16:08:06 2024
@author: Chen Huang <chen.huang23@imperial.ac.uk>
"""
from nanonis_tramea import Nanonis


class SemiqonLine:
    """
    Represents a single line in the Semiqon system with an optional label.

    Attributes:
        label (str): A descriptive label for the line.
    """

    def __init__(self, label: str = None):
        self.label = label


class SemiqonLinesConnection:
    """
    Defines a connection with labeled lines in the Semiqon system.

    This class holds a collection of `SemiqonLine` objects representing both
    the top and bottom lines in the connection.

    Attributes:
         lines (list of SemiqonLine): A list of SemiqonLine objects,
            each representing a line in the system.
    """

    def __init__(self):
        # Initialize lines with predefined labels for both top and bottom
        self.lines = [
            SemiqonLine(),  # empty
            # top lines
            SemiqonLine(label="t_D"),
            SemiqonLine(label="t_bar_4D"),
            SemiqonLine(label="t_P4"),
            SemiqonLine(label="t_bar_34"),
            SemiqonLine(label="t_P3"),
            SemiqonLine(label="t_bar_23"),
            SemiqonLine(label="t_P2"),
            SemiqonLine(label="t_bar_12"),
            SemiqonLine(label="t_P1"),
            SemiqonLine(label="t_bar_S1"),
            SemiqonLine(label="t_S"),
            SemiqonLine(label="res_S"),
            # bottom lines
            SemiqonLine(label="b_S"),
            SemiqonLine(label="b_bar_S1"),
            SemiqonLine(label="b_P1"),
            SemiqonLine(label="b_bar_12"),
            SemiqonLine(label="b_P2"),
            SemiqonLine(label="b_bar_23"),
            SemiqonLine(label="b_P3"),
            SemiqonLine(label="b_bar_34"),
            SemiqonLine(label="b_P4"),
            SemiqonLine(label="b_bar_4D"),
            SemiqonLine(label="b_D"),
            SemiqonLine(label="res_D"),
        ]


class NanonisSource:
    """
    Represents a source in the Nanonis system, capable of reading and setting voltage values.

    Attributes:
        label (str): A descriptive label for the source.
        read_index (int): The index used for reading from this source.
        write_index (int): The index used for writing to this source.
        nanonisInstance (Nanonis): An instance of the Nanonis class, representing the connection.
    """

    def __init__(
        self,
        label: str = None,
        read_index=None,
        write_index: int = None,
        nanonisInstance: Nanonis = None,
    ):
        self.label = label
        self.read_index = read_index
        self.write_index = write_index
        self.nanonisInstance = nanonisInstance


class NanonisSourceConnection:
    """
    Manages multiple sources in the Nanonis system, organizing them as inputs and outputs.

    Attributes:
        outputs (list of NanonisSource): A list of output sources, each with associated read/write indices.
        inputs (list of NanonisSource): A list of input sources, each with an associated read index.
    """

    def __init__(self, nanonisInstance: Nanonis = None):
        # Define outputs with specified labels and read/write indices
        self.outputs = [
            NanonisSource(),  # empty
            NanonisSource(
                label="Nanonis output1",
                read_index=24,
                write_index=1,
                nanonisInstance=nanonisInstance,
            ),
            NanonisSource(
                label="Nanonis output2",
                read_index=25,
                write_index=2,
                nanonisInstance=nanonisInstance,
            ),
            NanonisSource(
                label="Nanonis output3",
                read_index=26,
                write_index=3,
                nanonisInstance=nanonisInstance,
            ),
            NanonisSource(
                label="Nanonis output4",
                read_index=27,
                write_index=4,
                nanonisInstance=nanonisInstance,
            ),
            NanonisSource(
                label="Nanonis output5",
                read_index=28,
                write_index=5,
                nanonisInstance=nanonisInstance,
            ),
            NanonisSource(
                label="Nanonis output6",
                read_index=29,
                write_index=6,
                nanonisInstance=nanonisInstance,
            ),
            NanonisSource(
                label="Nanonis output7",
                read_index=30,
                write_index=7,
                nanonisInstance=nanonisInstance,
            ),
            NanonisSource(
                label="Nanonis output8",
                read_index=31,
                write_index=8,
                nanonisInstance=nanonisInstance,
            ),
        ]

        self.inputs = [
            NanonisSource(),  # empty
            NanonisSource(
                label="Nanonis input1", read_index=0, nanonisInstance=nanonisInstance
            ),
            NanonisSource(
                label="Nanonis input2", read_index=1, nanonisInstance=nanonisInstance
            ),
            NanonisSource(
                label="Nanonis input3", read_index=2, nanonisInstance=nanonisInstance
            ),
            NanonisSource(
                label="Nanonis input4", read_index=3, nanonisInstance=nanonisInstance
            ),
            NanonisSource(
                label="Nanonis input5", read_index=4, nanonisInstance=nanonisInstance
            ),
            NanonisSource(
                label="Nanonis input6", read_index=5, nanonisInstance=nanonisInstance
            ),
            NanonisSource(
                label="Nanonis input7", read_index=6, nanonisInstance=nanonisInstance
            ),
            NanonisSource(
                label="Nanonis input8", read_index=7, nanonisInstance=nanonisInstance
            ),
        ]
