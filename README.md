
# Monte Carlo Project

An educational project exploring Monte Carlo methods and GPU-accelerated computation with Python.

## Overview

Monte Carlo algorithms use repeated random sampling to estimate numerical results. This project compares a CPU implementation with a GPU implementation to demonstrate the performance benefits of parallel computation.

## Requirements

- Python 3.8 or newer
- The dependencies listed by the project
- A CUDA-compatible GPU and CUDA toolkit for GPU execution

## Getting started

1. Create and activate a virtual environment.
2. Install the project dependencies.
3. Run the provided Python program or notebook.

For example:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the CPU and GPU versions with the commands or notebooks included in the project. Adjust the number of samples to compare accuracy and execution time.

## Project structure

- CPU implementation: sequential Monte Carlo calculation
- GPU implementation: parallel Monte Carlo calculation
- Results: timing and accuracy comparison

## Notes

Results depend on the number of samples, random seed, hardware, and GPU configuration. GPU execution requires compatible drivers and libraries.

## License

This project is intended for educational use.
