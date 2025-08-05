"""
Cell Genome - Genetic encoding and decoding for cellular automaton

This module handles the bit-packed representation of cellular DNA,
including emission patterns and behavioral traits.
"""

import numpy as np
from typing import List

diagonal_interpolation = 0.5

class CellGenome:
    """
    Represents a cell's genetic code with 4-bit emission values for 4 directions
    and various behavioral parameters.
    
    Genome structure:
    - 4 emission directions (4 bits each) = 16 bits total
    - Reproduction threshold (4 bits)
    - Death threshold (4 bits)
    - One free byte for future use (8 bits) (diagonal_interpolation in 2 bits?)
    
    Total: 32 bits (fits in 4 bytes or 1 RGBA channel)
    """

    # Direction constants (matching shader order when you do [x - 1] x 2, the diagonals are interpolated)
    DIRECTIONS = [
        'north', 'northeast', 'east', 'southeast',
        'south', 'southwest', 'west', 'northwest'
    ]

    def __init__(self,
                 emissions: List[int] = [4, 3, 4, 3],
                 reproduction_threshold: int = 8,
                 death_threshold: int = 12):
        """
        Initialize a cell genome.
        
        Args:
            emissions: List of 4 emission values (0-15), one per direction
            reproduction_threshold: Energy needed to reproduce (0-15)
            death_threshold: Energy level that causes death (0-15)
        """
        
        self.emissions = self._validate_4bit_list(emissions, 4)
        self.reproduction_threshold = self._validate_4bit(reproduction_threshold)
        self.death_threshold = self._validate_4bit(death_threshold)

    @staticmethod
    def _validate_4bit(value: int) -> int:
        """Ensure value fits in 4 bits (0-15)"""
        return max(0, min(15, int(value)))
    
    @staticmethod
    def _validate_4bit_list(values: List[int], expected_length: int) -> List[int]:
        """Validate a list of 4-bit values"""
        if len(values) != expected_length:
            raise ValueError(f"Expected {expected_length} values, got {len(values)}")
        return [CellGenome._validate_4bit(v) for v in values]
    
    def pack_to_rgba(self) -> np.ndarray:
        """
        Pack the genome into a single RGBA channel.
        
        Returns:
            A NumPy array of shape (1, 4) containing the packed genome.
        """
        genome = np.array([
            (self.emissions[0] << 4) | self.emissions[2], # North and South emissions
            (self.emissions[1] << 4) | self.emissions[3], # East and West emissions
            (self.death_threshold << 4) | self.reproduction_threshold, # Death and Reproduction thresholds
            15  # Placeholder byte, can be used for future extensions
        ], dtype=np.uint8).reshape((1, 4))

        return genome
    
    @classmethod
    def unpack_from_rgba(cls, rgba: np.ndarray) -> 'CellGenome':
        """
        Unpack a genome from an RGBA array.
        
        Args:
            rgba: A NumPy array of shape (1, 4) containing the packed genome.
        
        Returns:
            An instance of CellGenome with unpacked values.
        """
        if rgba.shape != (1, 4):
            raise ValueError("RGBA array must have shape (1, 4)")
        
        emissions = [
            (rgba[0, 0] & 0xF0) >> 4,   # North
            (rgba[0, 1] & 0xF0) >> 4,   # East
            (rgba[0, 0] & 0x0F),        # South
            (rgba[0, 1] & 0x0F)         # West
        ]
        
        reproduction_threshold = rgba[0, 2] & 0x0F
        death_threshold = (rgba[0, 2] & 0xF0) >> 4

        return cls(emissions, reproduction_threshold, death_threshold)
    
    def total_emission(self) -> int:
        """
        Calculate the total emission value across all directions.
        
        Returns:
            Total emission value (sum of all emissions).
        """
        total = sum(self.emissions)
        total += (self.emissions[0] + self.emissions[1]) * diagonal_interpolation  # Add Northeast
        total += (self.emissions[1] + self.emissions[2]) * diagonal_interpolation  # Add Southeast
        total += (self.emissions[2] + self.emissions[3]) * diagonal_interpolation  # Add Southwest
        total += (self.emissions[3] + self.emissions[0]) * diagonal_interpolation  # Add Northwest
        return int(total)
    
    def genetic_distance(self, other: 'CellGenome') -> float:
        """
        Calculate genetic distance to another genome.
        
        Args:
            other: Another CellGenome instance
        
        Returns:
            Genetic distance as a float from 0 (identical) to 1 (completely different)
        """
        if not isinstance(other, CellGenome):
            raise TypeError("Can only compare with another CellGenome")
        
        emissions_distance = sum(abs(a - b) for a, b in zip(self.emissions, other.emissions)) / 4 # Normalize by number of emissions, max 15 distance total
        reproduction_distance = abs(self.reproduction_threshold - other.reproduction_threshold)
        death_distance = abs(self.death_threshold - other.death_threshold)

        max_distance = 15 + 15 + 15  # Max possible distance

        return (emissions_distance + reproduction_distance + death_distance) / max_distance

    def __str__(self) -> str:
        """
        String representation of the genome.
        
        Returns:
            A string summarizing the genome's emissions and thresholds.
        """
        return (f"CellGenome(emissions={self.emissions}, "
                f"reproduction_threshold={self.reproduction_threshold}, "
                f"death_threshold={self.death_threshold})")

    def __repr__(self) -> str:
        return self.__str__()
