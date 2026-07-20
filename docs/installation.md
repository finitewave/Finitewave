# Installation

To install Finitewave, run:

```bash
pip install finitewave
```

This will install Finitewave as a Python package on your system.

## Other installation options

You can also do it from source - navigate to the root directory of the project and run:

```bash
python -m build
pip install dist/finitewave-<version>.whl
```

For development purposes, install in editable mode (changes apply immediately without reinstall):

```bash
pip install -e .
```

---

## Requirements

Finitewave requires the following minimal versions:

| Dependency      | Version* | Link |
|----------------|----------|------|
| ffmpeg-python  | 0.2.0    | https://pypi.org/project/ffmpeg-python/ |
| matplotlib     | 3.9.2    | https://pypi.org/project/matplotlib/ |
| natsort        | 8.4.0    | https://pypi.org/project/natsort/ |
| numba          | 0.60.0   | https://pypi.org/project/numba/ |
| numpy          | 1.26.4   | https://pypi.org/project/numpy/ |
| pyvista        | 0.44.1   | https://pypi.org/project/pyvista/ |
| scikit-image   | 0.24.0   | https://pypi.org/project/scikit-image/ |
| scipy          | 1.14.1   | https://pypi.org/project/scipy/ |
| tqdm           | 4.66.5   | https://pypi.org/project/tqdm/ |

\* minimal version