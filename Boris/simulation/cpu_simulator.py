"""
CPU Simulator - Main simulation engine for genetic cellular automaton

This module handles the core simulation logic including cell lifecycle,
energy management, genetic operations, and grid state management.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any

from models.cell_genome import CellGenome
from models.cell_state import CellState

diagonal_interpolation = 0.5

class Cell:
    """
    Represents a cell in the simulation with its genome and state.
    
    Attributes:
        genome: The genetic code of the cell.
        state: The current state of the cell.
    """

    def __init__(self, genome: Optional[CellGenome], state: CellState):
        self.genome = genome
        self.state = state

    def __repr__(self):
        return f"Cell(genome={self.genome}, state={self.state})"
    
    def copy(self) -> 'Cell':
        """
        Create a copy of the cell.
        
        Returns:
            Cell: A new Cell instance with the same genome and state.
        """
        return Cell(genome=CellGenome(
            emissions=[self.genome.emissions[i] for i in range(4)] if self.genome else [0, 0, 0, 0],
            reproduction_threshold=self.genome.reproduction_threshold if self.genome else 0,
            death_threshold=self.genome.death_threshold if self.genome else 0
        ) if self.genome else None , state=CellState(self.state.energy, self.state.emissions))

class GeneticCellularAutomaton:
    """
    Main CPU-based simulation engine for the genetic cellular automaton.
    
    Handles:
    - Grid management
    - Cell lifecycle (birth, death, reproduction)
    - Energy management
    - Genetic operations (mutation, crossover)
    - State management (updating cell states)
    - Neighbor interactions
    """

    # Direction vectors for 8-neighbor Moore neighborhood
    DIRECTIONS = [
        (-1, 0),   # North
        (-1, 1),   # Northeast
        (0, 1),    # East
        (1, 1),    # Southeast
        (1, 0),    # South
        (1, -1),   # Southwest
        (0, -1),   # West
        (-1, -1),  # Northwest
    ]

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the simulation grid and parameters.
        
        Args:
            config: Configuration dictionary
        """

        self.config = config.copy()

        self.grid_size = config.get("grid_size", (100, 100))
        self.wrap_edges = config.get("wrap_edges", True)
        self.cells = config.get("initial_cells", []).copy()
        self.emission_decay = config.get("emission_decay", 0.1)

        self.total_energy = 0

        # Initialize grid as a NumPy array - create unique cells for each position
        self.total_energy = config.get("initial_energy", 100) * self.grid_size[0] * self.grid_size[1]
        # Create grid with unique Cell objects to avoid shared state
        self.grid = np.empty(self.grid_size, dtype=object)
        for y in range(self.grid_size[1]):
            for x in range(self.grid_size[0]):
                self.grid[y, x] = Cell(genome=None, state=CellState(energy=config.get("initial_energy", 100), emissions=0))
        
        for (y, x), cell in self.cells:
            self.grid[y, x] = cell.copy()
            self.total_energy += cell.state.energy - config.get("initial_energy", 100)
        
        self.generation = 0
        self.population = len(self.cells)
        self.births = 0
        self.deaths = 0
    
    def get_neighbors(self, x: int, y: int) -> List[Cell]:
        """
        Get neighboring cells in the Moore neighborhood.
        
        Args:
            x: X coordinate of the cell
            y: Y coordinate of the cell
        Returns:
            List[Cell]: List of neighboring cells (with order: N, NE, E, SE, S, SW, W, NW)
        """

        neighbors = []
        
        for i, (dx, dy) in enumerate(self.DIRECTIONS):
            if self.wrap_edges:
                nx = (x + dx) % self.grid_size[0]
                ny = (y + dy) % self.grid_size[1]
            else:
                nx = x + dx
                ny = y + dy
                if nx < 0 or nx >= self.grid_size[0] or ny < 0 or ny >= self.grid_size[1]:
                    continue
            
            cell = self.grid[ny][nx]
            neighbors.append(cell)
        
        return neighbors
    
    def mutate_genome(self, genome: CellGenome, mutation_seed: int) -> CellGenome:
        """
        Create a mutated copy of a genome using deterministic mutations.
        
        Args:
            genome: Original genome
            mutation_seed: Seed for deterministic mutation
            
        Returns:
            Mutated genome
        """
        def deterministic_noise(seed: int, gene_type: str, index: int) -> float:
            """Generate deterministic noise value between 0 and 1"""
            hash_value = hash((seed, gene_type, index))
            return (abs(hash_value) % 1000) / 1000.0
        
        def deterministic_choice(seed: int, gene_type: str, index: int, choices: list):
            """Deterministic choice from list"""
            noise = deterministic_noise(seed, gene_type, index)
            return choices[int(noise * len(choices))]
        
        # Copy current genome
        new_emissions = genome.emissions.copy()
        new_repro = genome.reproduction_threshold
        new_death = genome.death_threshold
        
        # Mutate emissions deterministically
        for i in range(len(new_emissions)):
            noise = deterministic_noise(mutation_seed, 'emission', i)
            if noise < self.config.get("mutation_strength", 0.1):
                change = deterministic_choice(mutation_seed, 'emission_change', i, [-1, 0, 1])
                new_emissions[i] = max(0, min(15, new_emissions[i] + change))
        
        # Mutate traits deterministically
        if deterministic_noise(mutation_seed, 'reproduction_threshold', 0) < self.config.get("mutation_strength", 0.1) * 0.5:
            change = deterministic_choice(mutation_seed, 'repro_change', 0, [-1, 1])
            new_repro = max(1, min(15, new_repro + change))

        if deterministic_noise(mutation_seed, 'death_threshold', 0) < self.config.get("mutation_strength", 0.1) * 0.5:
            change = deterministic_choice(mutation_seed, 'death_change', 0, [-1, 1])
            new_death = max(0, min(15, new_death + change))
        
        return CellGenome(
            emissions=new_emissions,
            reproduction_threshold=new_repro,
            death_threshold=new_death
        )

    def reproduce_cell(self, x: int, y: int):
        """
        Attempt to reproduce a cell at (x, y).
        
        Args:
            x: X coordinate of the cell
            y: Y coordinate of the cell
        """
        cell: Cell = self.grid[y][x]
        if cell.genome:
            # Create a new cell with a copy of the genome
            mutation_seed = hash((x, y, self.generation))
            new_genome = self.mutate_genome(cell.genome, mutation_seed)
            new_state = CellState(energy=0, emissions=0)
            new_cell = Cell(genome=new_genome, state=new_state)

            # Place the new cell in an empty neighboring space
            neighbors = self.get_neighbors(x, y)
            filtered_neighbors = [n for n in neighbors if n.genome is None]
            max_neighbor = max(filtered_neighbors, key=lambda n: n.state.emissions, default=None)
            if max_neighbor:
                max_neighbor_index = neighbors.index(max_neighbor)
                new_x = (x + self.DIRECTIONS[max_neighbor_index][0]) % self.grid_size[0]
                new_y = (y + self.DIRECTIONS[max_neighbor_index][1]) % self.grid_size[1]
                new_cell.state.energy = max_neighbor.state.energy
                self.grid[new_y][new_x] = new_cell
                self.births += 1
                self.population += 1
    
    def kill_cell(self, x: int, y: int):
        """
        Kill a cell at (x, y).
        
        Args:
            x: X coordinate of the cell
            y: Y coordinate of the cell
        """
        cell = self.grid[y][x]
        if cell.genome:
            # Remove the cell from the grid
            self.grid[y][x] = Cell(genome=None, state=CellState(energy=cell.state.energy, emissions=0))
            self.deaths += 1
            self.population -= 1

    def emission_phase(self):
        """
        Perform the emission phase where cells emit based on their genome and get emission decay.
        
        Each cell emits in its defined directions, which is then accumulated by neighboring cells.
        """
        for y in range(self.grid_size[1]):
            for x in range(self.grid_size[0]):
                cell: Cell = self.grid[y][x]

                # Decay emissions received from previous generations
                cell.state.emissions = int(cell.state.emissions * (1 - self.emission_decay))

                # Emit based on genome
                if cell.genome:
                    emissions = cell.genome.emissions
                    expanded_emissions = [
                        emissions[0],
                        min(int((emissions[0] + emissions[1]) * diagonal_interpolation), 15),
                        emissions[1],
                        min(int((emissions[1] + emissions[2]) * diagonal_interpolation), 15),
                        emissions[2],
                        min(int((emissions[2] + emissions[3]) * diagonal_interpolation), 15),
                        emissions[3],
                        min(int((emissions[3] + emissions[0]) * diagonal_interpolation), 15)
                    ]
                    for i, neighbor in enumerate(self.get_neighbors(x, y)):
                        neighbor.state.emissions = min(
                            neighbor.state.emissions + expanded_emissions[i],
                            255  # Cap emissions received at 255
                        )
                    # Decrease energy based on emissions
                    cell.state.energy = max(0, cell.state.energy - max(int(cell.genome.total_emission() * self.config.get("emission_cost", 0.1)), 1))

    def reaction_phase(self):
        """
        Perform the reaction phase where cells react to emissions from neighbors.
        """
        for y in range(self.grid_size[1]):
            for x in range(self.grid_size[0]):
                cell: Cell = self.grid[y][x]
                if cell.genome:
                    if cell.genome.reproduction_threshold * 16 <= cell.state.emissions:
                        self.reproduce_cell(x, y)
                    if cell.genome.death_threshold * 16 <= cell.state.emissions or cell.state.energy == 0 or cell.genome.total_emission() == 0:
                        self.kill_cell(x, y)

    def energy_phase(self):
        """
        Perform the energy phase where cells gain energy.
        
        Cells gain energy from net input to the world
        """
        self.total_energy += self.config.get("net_energy_input", 1) * self.grid_size[0] * self.grid_size[1]
        for y in range(self.grid_size[1]):
            for x in range(self.grid_size[0]):
                cell: Cell = self.grid[y][x]
                if cell.genome == None:
                    cell.state.energy = min(
                        self.config.get("net_energy_input", 1) + cell.state.energy,
                        255
                    )

    def step(self):
        """
        Perform a single simulation step.
        
        This includes emission, reaction, and energy phases.
        """
        self.births = 0
        self.deaths = 0
        self.emission_phase()
        self.reaction_phase()
        self.energy_phase()

        self.generation += 1