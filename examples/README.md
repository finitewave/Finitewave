# Finitewave Examples

This directory contains a collection of **example scripts** demonstrating how to use the Finitewave framework for cardiac electrophysiology simulations.

The examples are organized into subdirectories by topic. They cover a range of use cases — from basic functionality to advanced simulation setups.

## Structure

### 📁 `basics/`

Examples of **basic framework usage** and common cardiac phenomena:

- How to initialize and run 2D and 3D simulations
- Visualization of wave propagation
- Modeling of typical phenomena such as **spiral waves/reentry**

### 📁 `fibrosis/`

Examples of **simulations in fibrotic tissue**:

- Preparing fibrosis maps
- Studying wave behavior in heterogeneous tissue

### 📁 `models/`

**Minimal working examples** for each of the **electrophysiological models** implemented in Finitewave:

- Demonstrate basic usage of each model in isolation

### 📁 `stimulation/`

Examples of different **stimulation protocols**:

- stimulation by current/voltage
- stimulation by coordinates, matrices
- making stimulation sequences

### 📁 `trackers/`

Examples of using **trackers** included in the framework:

- How to measure activation times, APD, egm, period maps, etc.
- How to record and analyze simulation results during runtime

## How to run

You can run any example by executing it as a Python script:

```bash
python examples/<subdir>/<example_script.py>
