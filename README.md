# Homework 3 – JIT Compilation and Tracing

This repository contains my solutions for Homework 3.

## Contents
- `notebooks/jax_jit_analysis.ipynb` – JAX experiments
- `notebooks/torch_compile_analysis.ipynb` – PyTorch experiments

## Requirements
Install the required packages:

pip install jax jaxlib torch torchvision matplotlib

## Running the Code
Open the notebooks and run all cells:

Run:

- `jax_jit_analysis.ipynb`
- `torch_compile_analysis.ipynb`

These notebooks reproduce the experiments, plots, and results used in the report.

## Summary
The experiments analyze:

- JAX JIT compilation overhead
- Shape specialization in JAX
- Operator fusion benefits
- PyTorch `torch.compile` backends
- Graph capture and tracing with FX
