"""
Cell State - Represents the state of a cell in the simulation

This module defines the CellState class, which encapsulates the current
state of a cell, including energy level, accumulated emissions, and other states to be defined.
"""

import numpy as np

class CellState:
    """
    Represents the state of a cell in the simulation.
    
    State structure:
    - Energy level (0-255) (8 bits)
    - Emissions (0-255) (8 bits)
    - Other states to be added in the future (e.g. age, countdowns, etc.)
    """

    def __init__(self, energy: int = 0, emissions: int = 0):
        """
        Initialize a cell state.
        
        Args:
            energy: Initial energy level of the cell.
            emissions: Accumulated emissions (default is empty).
        """
        self.energy = max(0, min(int(energy), 255))  # Ensure 8-bit energy
        self.emissions = max(0, min(int(emissions), 255))  # Ensure 8-bit emissions

    def pack_to_rgba(self) -> np.ndarray:
        """
        Pack the cell state into a 4-byte RGBA format.
        
        Returns:
            np.ndarray: Array of shape (1, 4) containing the RGBA values.
        """
        rgba = np.array([
            self.energy, 
            self.emissions, 
            0,  # Placeholder for future states
            255  # Alpha channel (fully opaque), placeholder for future use
        ], dtype=np.uint8).reshape((1, 4))

        return rgba
    
    @classmethod
    def unpack_from_rgba(cls, rgba: np.ndarray) -> 'CellState':
        """
        Unpack a cell state from an RGBA array.
        
        Args:
            rgba: A NumPy array of shape (1, 4) containing the packed cell state.
        
        Returns:
            CellState: An instance of CellState with unpacked values.
        """
        if rgba.shape != (1, 4):
            raise ValueError("RGBA array must have shape (1, 4)")

        energy = rgba[0, 0]
        emissions = rgba[0, 1]

        return cls(energy=energy, emissions=emissions)

    def __str__(self) -> str:
        """
        String representation of the cell state.
        
        Returns:
            str: Formatted string showing energy and emissions.
        """
        return f"CellState(energy={self.energy}, emissions={self.emissions})"

    def __repr__(self) -> str:
        return self.__str__()