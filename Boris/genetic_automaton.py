"""
Genetic Cellular Automaton - Main Application

A Conway's Game of Life implementation with evolutionary genetics.
This is the main entry point that runs the CPU simulation with Pygame visualization.
"""

import pygame
import numpy as np
from typing import Any, Dict, Optional, Tuple, List

from models.cell_genome import CellGenome
from models.cell_state import CellState
from simulation.cpu_simulator import GeneticCellularAutomaton, Cell

Config = {
    "grid_size": (100, 100),    # Default grid size
    "initial_energy": 100,      # Initial energy for new cells
    "wrap_edges": True,         # Whether to wrap edges for neighbor calculations
    "initial_cells": [],        # Initial cells to populate the grid
    "emission_decay": 0.1,      # Decay rate for emissions
    "emission_cost": 0.1,       # Energy cost for emissions
    "net_energy_input": 1,      # Net energy input to the world
    "cell_size": 4,             # Size of each cell in pixels
    "mutation_strength": 0.1,   # Strength of mutation effects
}

def test_scenario_1() -> Dict[str, Any]:
    """
    Test scenario for the genetic cellular automaton.
    Tests a "construction" cell, a cell without emission (or very low emission) that, unless killed, lives forever.

    Returns:
        Dict[str, Any]: Configuration for the test scenario.
    """
    out = {
        "grid_size": (10, 10), 
        "initial_energy": 100,
        "wrap_edges": True,
        "initial_cells": [],
        "emission_decay": 0,
        "emission_cost": 0.1,
        "net_energy_input": 0,
        "cell_size": 20,
        "mutation_strength": 0
    }
    initial_cell = Cell(
        genome=CellGenome(
            emissions= [1, 1, 1, 1],
            reproduction_threshold=15,
            death_threshold=15,
        ),
        state=CellState(energy=100, emissions=0)
    )
    out["initial_cells"] = [((5, 5), initial_cell)]
    return out

def test_scenario_2() -> Dict[str, Any]:
    """
    Test scenario for the genetic cellular automaton.
    Test that a cell can die if it depletes it's energy
    
    Returns:
        Dict[str, Any]: Configuration for the test scenario.
    """
    out = {
        "grid_size": (10, 10), 
        "initial_energy": 100,
        "wrap_edges": True,
        "initial_cells": [],
        "emission_decay": 0,
        "emission_cost": 0.1,
        "net_energy_input": 0,
        "cell_size": 20,
        "mutation_strength": 0
    }
    initial_cell = Cell(
        genome=CellGenome(
            emissions= [10, 10, 0, 0],
            reproduction_threshold=15,
            death_threshold=15,
        ),
        state=CellState(energy=100, emissions=0)
    )
    out["initial_cells"] = [((5, 5), initial_cell)]
    return out

def test_scenario_3() -> Dict[str, Any]:
    """
    Test scenario for the genetic cellular automaton.
    Test that a cell can reproduce and die because of emissions.
    
    Returns:
        Dict[str, Any]: Configuration for the test scenario.
    """
    out = {
        "grid_size": (10, 10), 
        "initial_energy": 100,
        "wrap_edges": True,
        "initial_cells": [],
        "emission_decay": 0,
        "emission_cost": 0,
        "net_energy_input": 0,
        "cell_size": 20,
        "mutation_strength": 0
    }
    initial_cell1 = Cell(
        genome=CellGenome(
            emissions= [1, 0, 0, 0],
            reproduction_threshold=2,
            death_threshold=15,
        ),
        state=CellState(energy=100, emissions=0)
    )
    initial_cell2 = Cell(
        genome=CellGenome(
            emissions= [15, 0, 0, 0],
            reproduction_threshold=15,
            death_threshold=15,
        ),
        state=CellState(energy=100, emissions=0)
    )
    out["initial_cells"] = [((5, 5), initial_cell1), ((5, 6), initial_cell2)]
    return out

def test_scenario_4() -> Dict[str, Any]:
    """
    Test scenario for the genetic cellular automaton.
    Test a more complex scenario with one kind of cell that travels northward, relying on natural energy input.
    
    Returns:
        Dict[str, Any]: Configuration for the test scenario.
    """
    out = {
        "grid_size": (100, 100), 
        "initial_energy": 100,
        "wrap_edges": True,
        "initial_cells": [],
        "emission_decay": 0.01,
        "emission_cost": 1,
        "net_energy_input": 10,
        "cell_size": 5,
        "mutation_strength": 0
    }
    initial_cell = Cell(
        genome=CellGenome(
            emissions= [6, 0, 2, 0],
            reproduction_threshold=1,
            death_threshold=2,
        ),
        state=CellState(energy=100, emissions=0)
    )
    out["initial_cells"] = [((5, 5), initial_cell), ((5, 6), initial_cell), ((6, 5), initial_cell), ((6, 6), initial_cell)]
    return out

def test_scenario_5() -> Dict[str, Any]:
    """
    Test scenario for the genetic cellular automaton.
    Test a more complex scenario with one kind of cell traveling north-east and produces generative vortexes.
    
    Returns:
        Dict[str, Any]: Configuration for the test scenario.
    """
    out = {
        "grid_size": (100, 100), 
        "initial_energy": 100,
        "wrap_edges": True,
        "initial_cells": [],
        "emission_decay": 0.05,
        "emission_cost": 1,
        "net_energy_input": 8,
        "cell_size": 5,
        "mutation_strength": 0
    }
    initial_cell = Cell(
        genome=CellGenome(
            emissions= [4, 4, 1, 1],
            reproduction_threshold=1,
            death_threshold=2,
        ),
        state=CellState(energy=100, emissions=0)
    )
    out["initial_cells"] = [((5, 5), initial_cell), ((5, 6), initial_cell), ((6, 5), initial_cell), ((6, 6), initial_cell)]
    return out

def test_scenario_6() -> Dict[str, Any]:
    """
    Test scenario for the genetic cellular automaton.
    Test a more scenario with two kinds of cells that battle. It has some time with coexistence, but eventually one of the two kinds of cells wins.
    
    Returns:
        Dict[str, Any]: Configuration for the test scenario.
    """
    out = {
        "grid_size": (70, 70), 
        "initial_energy": 100,
        "wrap_edges": True,
        "initial_cells": [],
        "emission_decay": 0.1,
        "emission_cost": 1,
        "net_energy_input": 8,
        "cell_size": 5,
        "mutation_strength": 0
    }
    initial_cell1 = Cell(
        genome=CellGenome(
            emissions= [4, 4, 1, 1],
            reproduction_threshold=1,
            death_threshold=2,
        ),
        state=CellState(energy=100, emissions=0)
    )
    initial_cell2 = Cell(
        genome=CellGenome(
            emissions= [1, 4, 4, 1],
            reproduction_threshold=1,
            death_threshold=2,
        ),
        state=CellState(energy=100, emissions=0)
    )
    out["initial_cells"] = [((5, 5), initial_cell1), ((5, 6), initial_cell1), ((6, 5), initial_cell1), ((6, 6), initial_cell1),
                            ((30, 30), initial_cell2), ((30, 31), initial_cell2), ((31, 30), initial_cell2), ((31, 31), initial_cell2)]
    return out

def test_scenario_7() -> Dict[str, Any]:
    """
    Test scenario for the genetic cellular automaton.
    Test a supr aggresive cell that expands a lot on high energy input, but dies quickly and travels on a light mode.
    
    Returns:
        Dict[str, Any]: Configuration for the test scenario.
    """
    out = {
        "grid_size": (100, 100), 
        "initial_energy": 150,
        "wrap_edges": True,
        "initial_cells": [],
        "emission_decay": 0.05,
        "emission_cost": 1,
        "net_energy_input": 1,
        "cell_size": 5,
        "mutation_strength": 0
    }
    initial_cell = Cell(
        genome=CellGenome(
            emissions= [12, 2, 2, 12],
            reproduction_threshold=1,
            death_threshold=2,
        ),
        state=CellState(energy=100, emissions=0)
    )
    out["initial_cells"] = [((5, 5), initial_cell), ((5, 6), initial_cell), ((6, 5), initial_cell), ((6, 6), initial_cell)]
    return out

def test_scenario_8() -> Dict[str, Any]:
    """
    Test scenario for the genetic cellular automaton.
    Test a more scenario with two kinds of cells that battle, apexPredator vs ultraPest
    
    Returns:
        Dict[str, Any]: Configuration for the test scenario.
    """
    out = {
        "grid_size": (150, 150), 
        "initial_energy": 75,
        "wrap_edges": True,
        "initial_cells": [],
        "emission_decay": 0.05,
        "emission_cost": 1,
        "net_energy_input": 2,
        "cell_size": 5,
        "mutation_strength": 0
    }
    initial_cell2 = Cell(
        genome=CellGenome(
            emissions= [3, 3, 3, 3],
            reproduction_threshold=1,
            death_threshold=15,
        ),
        state=CellState(energy=100, emissions=0)
    )
    initial_cell1 = Cell(
        genome=CellGenome(
            emissions= [14, 2, 2, 14],
            reproduction_threshold=1,
            death_threshold=15,
        ),
        state=CellState(energy=100, emissions=0)
    )
    out["initial_cells"] = [((5, 5), initial_cell1), ((5, 6), initial_cell1), ((6, 5), initial_cell1), ((6, 6), initial_cell1),
                            ((30, 30), initial_cell2), ((30, 31), initial_cell2), ((31, 30), initial_cell2), ((31, 31), initial_cell2)]
    return out

def test_scenario_9() -> Dict[str, Any]:
    """
    Test scenario for the genetic cellular automaton.
    Test a ultra low cost cell, grass-like.
    
    Returns:
        Dict[str, Any]: Configuration for the test scenario.
    """
    out = {
        "grid_size": (70, 70), 
        "initial_energy": 50,
        "wrap_edges": True,
        "initial_cells": [],
        "emission_decay": 0.1,
        "emission_cost": 0.5,
        "net_energy_input": 5,
        "cell_size": 10,
        "mutation_strength": 0
    }
    initial_cell1 = Cell(
        genome=CellGenome(
            emissions= [1, 1, 1, 1],
            reproduction_threshold=1,
            death_threshold=2,
        ),
        state=CellState(energy=100, emissions=0)
    )
    out["initial_cells"] = [((5, 5), initial_cell1), ((5, 6), initial_cell1), ((6, 5), initial_cell1), ((6, 6), initial_cell1)]
    return out

def test_scenario_10() -> Dict[str, Any]:
    """
    Test scenario for the genetic cellular automaton.
    Test apexPredator vs lowCost cell, lowCost ends up winning after many generations.
    
    Returns:
        Dict[str, Any]: Configuration for the test scenario.
    """
    out = {
        "grid_size": (70, 70), 
        "initial_energy": 25,
        "wrap_edges": True,
        "initial_cells": [],
        "emission_decay": 0.1,
        "emission_cost": 0.5,
        "net_energy_input": 5,
        "cell_size": 10,
        "mutation_strength": 0.01
    }
    initial_cell1 = Cell(
        genome=CellGenome(
            emissions= [14, 2, 2, 14],
            reproduction_threshold=1,
            death_threshold=15,
        ),
        state=CellState(energy=100, emissions=0)
    )
    initial_cell2 = Cell(
        genome=CellGenome(
            emissions= [1, 1, 1, 1],
            reproduction_threshold=1,
            death_threshold=2,
        ),
        state=CellState(energy=100, emissions=0)
    )
    out["initial_cells"] = [((5, 5), initial_cell1), ((5, 6), initial_cell1), ((6, 5), initial_cell1), ((6, 6), initial_cell1),
                            ((30, 30), initial_cell2), ((30, 31), initial_cell2), ((31, 30), initial_cell2), ((31, 31), initial_cell2)]
    return out

def random_scenario(
    grid_size: Optional[Tuple[int, int]] = None,
    num_species: Optional[int] = None,
    cluster_size: Optional[int] = None,
    mutation_variance: Optional[float] = None,
    energy_level: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a random configuration with diverse cell types grouped in clusters.
    All parameters are randomized if not specified.
    
    Args:
        grid_size: Size of the simulation grid (randomized if None)
        num_species: Number of different cell species to create (randomized if None)
        cluster_size: Approximate size of each species cluster (randomized if None)
        mutation_variance: How much variation to allow in genome parameters (randomized if None)
        energy_level: "low", "medium", or "high" energy environment (randomized if None)
    
    Returns:
        Dict[str, Any]: Configuration for the random scenario.
    """
    import random
    
    # Randomize parameters if not provided
    if grid_size is None:
        # Random grid size between 50x50 and 100x100
        size = random.randint(50, 100)
        grid_size = (size, size)
    
    if num_species is None:
        # Random number of species between 3 and 8
        num_species = random.randint(3, 8)
    
    if cluster_size is None:
        # Random cluster size between 2 and 6
        cluster_size = random.randint(2, 6)
    
    if mutation_variance is None:
        # Random mutation variance between 0.5 and 4.0
        mutation_variance = random.uniform(0.5, 4.0)
    
    if energy_level is None:
        # Random energy level
        energy_level = random.choice(["low", "medium", "high"])
    
    # Energy configurations based on level (now with some randomization too)
    energy_configs = {
        "low": {
            "initial_energy": random.randint(30, 70),
            "emission_cost": random.uniform(0.8, 1.5),
            "net_energy_input": random.randint(1, 4),
            "emission_decay": random.uniform(0.05, 0.15)
        },
        "medium": {
            "initial_energy": random.randint(70, 130),
            "emission_cost": random.uniform(0.4, 1.0),
            "net_energy_input": random.randint(3, 8),
            "emission_decay": random.uniform(0.02, 0.08)
        },
        "high": {
            "initial_energy": random.randint(120, 200),
            "emission_cost": random.uniform(0.1, 0.6),
            "net_energy_input": random.randint(6, 15),
            "emission_decay": random.uniform(0.01, 0.05)
        }
    }
    
    energy_config = energy_configs[energy_level]
    
    out = {
        "grid_size": grid_size,
        "initial_energy": energy_config["initial_energy"],
        "wrap_edges": random.choice([True, False]),  # Randomize edge wrapping
        "initial_cells": [],
        "emission_decay": energy_config["emission_decay"],
        "emission_cost": energy_config["emission_cost"],
        "net_energy_input": energy_config["net_energy_input"],
        "cell_size": min(15, max(2, min(1800 // grid_size[0], 1000 // grid_size[1]))),  # Auto-size for 1080p screen
        "mutation_strength": random.uniform(0.001, 0.05)  # Random mutation strength
    }
    
    # Generate diverse base genome archetypes (now with some randomization)
    base_archetypes = [
        # Aggressive predator - high emissions, low thresholds
        {
            "emissions": [12, 3, 3, 12],
            "reproduction_threshold": 1,
            "death_threshold": 2,
            "description": "aggressive_predator"
        },
        # Balanced omnivore - moderate everything
        {
            "emissions": [6, 6, 6, 6],
            "reproduction_threshold": 3,
            "death_threshold": 8,
            "description": "balanced_omnivore"
        },
        # Conservative survivor - low emissions, high death threshold
        {
            "emissions": [2, 2, 2, 2],
            "reproduction_threshold": 5,
            "death_threshold": 12,
            "description": "conservative_survivor"
        },
        # Directional traveler - focused emissions in one direction
        {
            "emissions": [15, 0, 3, 0],
            "reproduction_threshold": 2,
            "death_threshold": 4,
            "description": "directional_traveler"
        },
        # Rapid reproducer - very low reproduction threshold
        {
            "emissions": [4, 4, 4, 4],
            "reproduction_threshold": 1,
            "death_threshold": 3,
            "description": "rapid_reproducer"
        },
        # Tank - high death threshold, moderate emissions
        {
            "emissions": [8, 2, 8, 2],
            "reproduction_threshold": 6,
            "death_threshold": 15,
            "description": "tank"
        },
        # Efficient - very low emission cost
        {
            "emissions": [1, 1, 1, 1],
            "reproduction_threshold": 1,
            "death_threshold": 2,
            "description": "efficient"
        }
    ]
    
    # Add some completely random archetypes for variety
    random_archetypes = []
    for _ in range(random.randint(1, 3)):  # Add 1-3 completely random archetypes
        random_archetypes.append({
            "emissions": [random.randint(0, 15) for _ in range(4)],
            "reproduction_threshold": random.randint(1, 8),  # Ensure never 0
            "death_threshold": random.randint(1, 15),
            "description": "random_mutation"
        })
    
    # Combine base archetypes with random ones
    all_archetypes = base_archetypes + random_archetypes
    
    # Select random archetypes for this scenario
    selected_archetypes = random.sample(all_archetypes, min(num_species, len(all_archetypes)))
    
    # Calculate positions for clusters with more randomization
    cluster_positions = []
    grid_w, grid_h = grid_size
    
    # Choose between different positioning strategies
    positioning_strategy = random.choice(["grid", "random", "corners", "edges"])
    
    if positioning_strategy == "grid":
        # Original grid-based positioning with more randomness
        spacing_x = grid_w // (int(np.sqrt(num_species)) + 1)
        spacing_y = grid_h // (int(np.sqrt(num_species)) + 1)
        
        for i in range(num_species):
            grid_x = i % int(np.sqrt(num_species) + 1)
            grid_y = i // int(np.sqrt(num_species) + 1)
            
            center_x = spacing_x * (grid_x + 1) + random.randint(-spacing_x//2, spacing_x//2)
            center_y = spacing_y * (grid_y + 1) + random.randint(-spacing_y//2, spacing_y//2)
            
            center_x = max(cluster_size, min(grid_w - cluster_size, center_x))
            center_y = max(cluster_size, min(grid_h - cluster_size, center_y))
            
            cluster_positions.append((center_x, center_y))
    
    elif positioning_strategy == "random":
        # Completely random positioning
        for _ in range(num_species):
            center_x = random.randint(cluster_size, grid_w - cluster_size)
            center_y = random.randint(cluster_size, grid_h - cluster_size)
            cluster_positions.append((center_x, center_y))
    
    elif positioning_strategy == "corners":
        # Place species near corners and center
        possible_positions = [
            (cluster_size * 2, cluster_size * 2),  # Top-left
            (grid_w - cluster_size * 2, cluster_size * 2),  # Top-right
            (cluster_size * 2, grid_h - cluster_size * 2),  # Bottom-left
            (grid_w - cluster_size * 2, grid_h - cluster_size * 2),  # Bottom-right
            (grid_w // 2, grid_h // 2),  # Center
            (grid_w // 4, grid_h // 2),  # Left-center
            (3 * grid_w // 4, grid_h // 2),  # Right-center
            (grid_w // 2, grid_h // 4),  # Top-center
            (grid_w // 2, 3 * grid_h // 4),  # Bottom-center
        ]
        selected_positions = random.sample(possible_positions, min(num_species, len(possible_positions)))
        cluster_positions = selected_positions
    
    elif positioning_strategy == "edges":
        # Place species along the edges
        for i in range(num_species):
            edge = random.choice(["top", "bottom", "left", "right"])
            if edge == "top":
                center_x = random.randint(cluster_size, grid_w - cluster_size)
                center_y = cluster_size * 2
            elif edge == "bottom":
                center_x = random.randint(cluster_size, grid_w - cluster_size)
                center_y = grid_h - cluster_size * 2
            elif edge == "left":
                center_x = cluster_size * 2
                center_y = random.randint(cluster_size, grid_h - cluster_size)
            else:  # right
                center_x = grid_w - cluster_size * 2
                center_y = random.randint(cluster_size, grid_h - cluster_size)
            
            cluster_positions.append((center_x, center_y))
    
    # Generate cells for each species
    for i, archetype in enumerate(selected_archetypes):
        if i >= len(cluster_positions):
            break
            
        center_x, center_y = cluster_positions[i]
        
        # Add some mutation to the base archetype
        mutated_emissions = []
        for emission in archetype["emissions"]:
            variance = max(1, int(mutation_variance))
            mutated_value = emission + random.randint(-variance, variance)
            mutated_emissions.append(max(0, min(15, mutated_value)))
        
        reproduction_variance = max(1, int(mutation_variance))
        death_variance = max(1, int(mutation_variance))
        
        mutated_reproduction = max(1, min(15,  # Ensure never 0, minimum is 1
            archetype["reproduction_threshold"] + random.randint(-reproduction_variance, reproduction_variance)))
        mutated_death = max(0, min(15, 
            archetype["death_threshold"] + random.randint(-death_variance, death_variance)))
        
        # Create the species cell
        species_cell = Cell(
            genome=CellGenome(
                emissions=mutated_emissions,
                reproduction_threshold=mutated_reproduction,
                death_threshold=mutated_death,
            ),
            state=CellState(energy=energy_config["initial_energy"], emissions=0)
        )
        
        # Place cluster around the center position with random shape
        cluster_radius = cluster_size // 2
        cluster_shape = random.choice(["square", "circle", "cross", "diamond", "scattered"])
        fill_density = random.uniform(0.5, 0.9)  # Random fill density
        
        for dx in range(-cluster_radius, cluster_radius + 1):
            for dy in range(-cluster_radius, cluster_radius + 1):
                # Apply shape constraints
                place_cell = False
                
                if cluster_shape == "square":
                    place_cell = True
                elif cluster_shape == "circle":
                    distance = np.sqrt(dx*dx + dy*dy)
                    place_cell = distance <= cluster_radius
                elif cluster_shape == "cross":
                    place_cell = (dx == 0) or (dy == 0)
                elif cluster_shape == "diamond":
                    place_cell = abs(dx) + abs(dy) <= cluster_radius
                elif cluster_shape == "scattered":
                    # More random, scattered placement
                    distance = np.sqrt(dx*dx + dy*dy)
                    place_cell = distance <= cluster_radius * random.uniform(0.8, 1.2)
                
                # Apply density constraint
                if place_cell and random.random() < fill_density:
                    x = center_x + dx
                    y = center_y + dy
                    
                    # Make sure we're within bounds
                    if 0 <= x < grid_w and 0 <= y < grid_h:
                        # Add slight individual mutations to each cell
                        individual_cell = Cell(
                            genome=CellGenome(
                                emissions=[max(0, min(15, e + random.randint(-1, 1))) for e in mutated_emissions],
                                reproduction_threshold=max(1, min(15, mutated_reproduction + random.randint(-1, 1))),  # Ensure never 0
                                death_threshold=max(0, min(15, mutated_death + random.randint(-1, 1))),
                            ),
                            state=CellState(energy=energy_config["initial_energy"], emissions=0)
                        )
                        out["initial_cells"].append(((x, y), individual_cell))
    
    return out

def random_battle_scenario() -> Dict[str, Any]:
    """Generate a random battle scenario with 3-4 competing species."""
    return random_scenario(
        grid_size=(80, 80),
        num_species=4,
        cluster_size=3,
        mutation_variance=1.5,
        energy_level="medium"
    )

def random_survival_scenario() -> Dict[str, Any]:
    """Generate a harsh survival scenario with limited energy."""
    return random_scenario(
        grid_size=(100, 100),
        num_species=6,
        cluster_size=2,
        mutation_variance=2.0,
        energy_level="low"
    )

def random_abundance_scenario() -> Dict[str, Any]:
    """Generate an abundant energy scenario for rapid evolution."""
    return random_scenario(
        grid_size=(120, 120),
        num_species=7,
        cluster_size=5,
        mutation_variance=3.0,
        energy_level="high"
    )

class GeneticAutomatonApp:
    """
    Main application class that handles Pygame visualization and user interaction.
    """

    def __init__(self, config: Dict[str, Any] = Config):
        self.config = config
        self.original_config = config.copy()
        self.cell_size = config["cell_size"]
        self.simulator = GeneticCellularAutomaton(config)
        self.screen = pygame.display.set_mode((config["grid_size"][0] * self.cell_size, config["grid_size"][1] * self.cell_size))
        self.clock = pygame.time.Clock()

        pygame.display.set_caption("Genetic Cellular Automaton")
        pygame.init()

        self.running = True
        self.paused = False
        self.frame_count = 0
        self.target_fps = 10
    
    def get_color_cell(self, cell: Cell) -> Tuple[int, int, int]:
        """
        Get the color representation of a cell based on its genome.
        
        Args:
            cell: The cell to get the color for.

        Returns:
            Tuple of RGB values.
        """
        if cell.genome is None:
            return (0, 0, 0)
        else:
            """r = cell.genome.pack_to_rgba()[0, 0]
            g = cell.genome.pack_to_rgba()[0, 1]
            b = cell.genome.pack_to_rgba()[0, 2]
            """

            r = min(cell.genome.total_emission() * 8, 255)
            g = min(cell.genome.reproduction_threshold * 16, 255)
            b = min(cell.genome.death_threshold * 16, 255)

            return (r, g, b)
    
    def draw_grid(self):
        """
        Draw the grid of cells on the Pygame screen.
        """
        for y in range(self.config["grid_size"][1]):
            for x in range(self.config["grid_size"][0]):
                cell = self.simulator.grid[x][y]
                color = self.get_color_cell(cell)
                pygame.draw.rect(self.screen, color, (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))
    
    def draw_ui(self):
        """
        Draw the UI elements on the Pygame screen.
        """
        font = pygame.font.Font(None, 24)
        small_font = pygame.font.Font(None, 18)
        
        # Main info line
        info_text = f"Generation: {self.simulator.generation} | FPS: {self.clock.get_fps():.2f} | Population: {self.simulator.population} | Births: {self.simulator.births} | Deaths: {self.simulator.deaths}"
        text_surface = font.render(info_text, True, (255, 255, 255))
        self.screen.blit(text_surface, (10, 10))
        
        # Controls help
        controls_text = "Controls: SPACE=Pause/Resume | R=Reset | N=New Scenario | ESC=Quit | ↑↓=Cell Size | ←→=Speed | S=Step"
        controls_surface = small_font.render(controls_text, True, (200, 200, 200))
        self.screen.blit(controls_surface, (10, 35))
    
    def handle_events(self):
        """
        Handle Pygame events such as quitting and pausing the simulation.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                
                elif event.key == pygame.K_ESCAPE:
                    self.running = False
                
                elif event.key == pygame.K_r:
                    # Reset the simulation using original config
                    self.simulator = GeneticCellularAutomaton(self.original_config.copy())
                    self.frame_count = 0
                    # Update screen size in case cell_size changed
                    self.screen = pygame.display.set_mode((self.config["grid_size"][0] * self.cell_size, self.config["grid_size"][1] * self.cell_size))
                    # Clear screen and redraw
                    self.screen.fill((0, 0, 0))
                    self.update()
                
                elif event.key == pygame.K_n:
                    # Generate a new random scenario
                    new_config = random_scenario()
                    self.config = new_config
                    self.original_config = new_config.copy()
                    self.cell_size = new_config["cell_size"]
                    self.simulator = GeneticCellularAutomaton(new_config)
                    self.frame_count = 0
                    # Update screen size for new grid
                    self.screen = pygame.display.set_mode((new_config["grid_size"][0] * self.cell_size, new_config["grid_size"][1] * self.cell_size))
                    # Clear screen and redraw
                    self.screen.fill((0, 0, 0))
                    self.update()
                    print(f"Generated new scenario: {new_config['grid_size']} grid with {len(new_config['initial_cells'])} initial cells")
                
                elif event.key == pygame.K_UP:
                    self.cell_size += 1
                    # Resize screen immediately
                    self.screen = pygame.display.set_mode((self.config["grid_size"][0] * self.cell_size, self.config["grid_size"][1] * self.cell_size))

                elif event.key == pygame.K_DOWN:
                    self.cell_size = max(1, self.cell_size - 1)
                    # Resize screen immediately
                    self.screen = pygame.display.set_mode((self.config["grid_size"][0] * self.cell_size, self.config["grid_size"][1] * self.cell_size))
                
                elif event.key == pygame.K_LEFT:
                    # Decrease the target FPS
                    self.target_fps = max(1, self.target_fps - 5)
                
                elif event.key == pygame.K_RIGHT:
                    # Increase the target FPS
                    self.target_fps += 5

                elif event.key == pygame.K_s:
                    # Step the simulation manually
                    if self.paused:
                        self.update()
    
    def update(self):
        """
        Update the simulation state and redraw the screen.
        """
        self.frame_count += 1
        self.simulator.step()
        # Clear the screen first
        self.screen.fill((0, 0, 0))
        self.draw_grid()
        self.draw_ui()
        pygame.display.flip()

    def run(self):
        """
        Main loop to run the Pygame application.
        """
        while self.running:
            self.handle_events()

            if not self.paused:
                self.update()
            
            # Cap the frame rate
            self.clock.tick(self.target_fps)
        
        pygame.quit()

def main():
    """
    Main entry point for the application.
    """
    # Choose one of these scenarios:
    # app = GeneticAutomatonApp(test_scenario_3())  # Predefined test
    # app = GeneticAutomatonApp(random_battle_scenario())  # 4 species battle
    # app = GeneticAutomatonApp(random_survival_scenario())  # Harsh survival
    # app = GeneticAutomatonApp(random_abundance_scenario())  # High energy evolution
    app = GeneticAutomatonApp(random_scenario())  # Default random scenario
    
    app.paused = True  # Start paused for initial setup
    app.update()  # Draw initial state
    try:
        app.run()
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pygame.quit()

if __name__ == "__main__":
    main()