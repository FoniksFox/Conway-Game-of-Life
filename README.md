# Conway-Game-of-Life

A week-long project focused on building Conway's Game of Life with the Pygame library and, possibly, parallel processing or multithreading.

## What is Conway's Game of Life?

Conway's Game of Life is a cellular automaton devised by mathematician John Conway. It's a zero-player game where you create an initial configuration and observe how it evolves. The game consists of a grid of cells that can be either alive or dead, with simple rules determining their fate:

- **Underpopulation**: Any live cell with fewer than two live neighbors dies
- **Survival**: Any live cell with two or three live neighbors survives  
- **Overpopulation**: Any live cell with more than three live neighbors dies
- **Reproduction**: Any dead cell with exactly three live neighbors becomes alive

## Project Overview

This project is part of the Weekly-Projects series aimed at exploring different programming technologies via practical applications. The goal is to create a game that demonstrates understanding of Python's capabilities and, if desired, of parallel processing, multithreading or computer shaders.

## Learning Objectives

**Beginner Level**
- **Python Syntax** - Master basic Python programming concepts
- **Library Usage** - Learn to work with external libraries like Pygame

**Intermediate Level**
- **Algorithm Understanding** - Develop deeper comprehension of computational algorithms
- **Documentation** - Create clear, comprehensive project documentation
- **Version Control** - Practice Git and GitHub workflows

**Advanced Level**
- **Parallelization Techniques** - Implement CPU multithreading or GPU computing

## Project Structure

This is a collaborative project where multiple people work independently on different implementations of Conway's Game of Life.

**Implementation Options:**
- **Option 1 (Recommended)**: Create a folder and branch with your name for your implementation
- **Option 2**: Fork the repository and work on your own copy
- **Option 3**: Develop locally and share your results

**Requirements:**
- Each implementation should include a personal README explaining your approach
- Document your design decisions and any unique features you add

## Getting Started

### Prerequisites
- Python 3.7 or higher
- Basic understanding of Python programming

### Installation
```bash
# Clone the repository
git clone https://github.com/FoniksFox/Weekly-Projects.git
cd Weekly-Projects/Conway-Game-of-Life

# Create your implementation folder
mkdir YourName
cd YourName

# Install required dependencies
pip install pygame
# Add any other dependencies as needed
```

## Key Features

### Beginner Level - Core Features
- [ ] **Functional Display** - Visual grid showing cell states
- [ ] **Basic Controls** - Play, pause, and restart functionality  
- [ ] **Game Logic** - Correct implementation of Conway's rules

### Intermediate Level - Enhanced Features
- [ ] **Comprehensive Documentation** - Clear code comments and user guide
- [ ] **Version Control** - Proper Git workflow and commit history
- [ ] **Creative Twist** - Add unique features like:
  - Player interaction elements
  - Matrix multiplication-based algorithms
  - Genetic systems for cells
  - Custom rule sets
- [ ] **Advanced Controls** - Speed adjustment, configurable grid size, pattern loading
- [ ] **Performance Optimization** - Efficient algorithms beyond brute force approaches

### Advanced Level - High Performance
- [ ] **Parallel Computing** - CPU multithreading or GPU shaders for computation

## Technology Stack

### Core Technologies
- **Python 3.7+** - Primary programming language
- **Pygame** - Graphics and user interface library

### Optional Libraries
- **NumPy** - For matrix operations and performance optimization
- **Multiprocessing/Threading** - For parallel computation
- **OpenCV** - For advanced graphics processing
- **CUDA/OpenCL** - For GPU acceleration (advanced)

## Example Implementation Ideas

### Beginner Projects
- Basic Game of Life with simple grid display
- Save/load functionality for configurations

### Intermediate Projects  
- Interactive editor for creating initial patterns
- Performance metrics and statistics display
- Multiple simultaneous universes
- Custom rule variants (different neighbor counts, wraparound edges)

### Advanced Projects
- Real-time performance with large grids (1000x1000+)
- GPU-accelerated computation using shaders
- 3D Game of Life implementation
- Machine learning pattern recognition

## Success Criteria

- ✅ **Functional Implementation** - Working Game of Life following Conway's rules
- ✅ **Clean Code** - Well-structured, documented, and maintainable codebase  
- ✅ **Good Performance** - Smooth operation even on modest hardware
- ✅ **Documentation** - Clear README and code comments
- ✅ **Version Control** - Proper Git usage with meaningful commits

### Bonus Goals
- ✅ **Parallel Computing** - Multithreaded or GPU-accelerated computation
- ✅ **Creative Features** - Unique additions beyond the basic requirements
- ✅ **User Experience** - Intuitive controls and visual polish

## Resources

### Conway's Game of Life
- [Wikipedia - Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life)
- [LifeWiki - Comprehensive Game of Life resource](https://conwaylife.com/)
- [Interactive Game of Life simulator](https://playgameoflife.com/)

### Python and Pygame
- [Python Documentation](https://docs.python.org/3/)
- [Pygame Documentation](https://www.pygame.org/docs/)
- [Pygame CE Documentation](https://pyga.me/docs/) (Community Edition)

### Algorithms and Optimization
- [Game of Life Algorithms](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life#Algorithms)
- [Optimization Techniques for Cellular Automata](https://stackoverflow.com/questions/40485496/optimizing-conways-game-of-life)

### Parallel Computing
- [Python Multiprocessing Documentation](https://docs.python.org/3/library/multiprocessing.html)
- [NumPy for vectorized operations](https://numpy.org/doc/stable/)
- [CUDA Python for GPU computing](https://developer.nvidia.com/cuda-python)

## Contributing

1. **Fork or Branch** - Create your implementation space
2. **Document** - Include a README explaining your approach  
3. **Test** - Ensure your implementation works correctly
4. **Share** - Submit your work for others to learn from

*Ask anything you need, as stupid as it may seem, the objective here is to learn together!

---

## Project Information

- **Started:** August 04, 2025  
- **Main Maintainer:** Boris Mladenov Beslimov
- **Repository:** Weekly-Projects / Conway-Game-of-Life