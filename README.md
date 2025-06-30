# Compact-Kernel (CK) MPM 
![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

This repository contains an implementation of Compact Kernel Material Point Method (CK-MPM) simulator and a set of example tests.  The tests are built as standalone executables using
[CMake](https://cmake.org/) and require a working CUDA toolchain. The code is built and tested on Ubuntu.

## Prerequisites

- CMake **3.27** or newer
- A C++20 compatible compiler with CUDA support (tested with the NVIDIA CUDA
  Toolkit)
- `libglut-dev`
- `python3-dev`
- Git & Git-Lfs

## Cloning the Repository

This project uses Git submodules for its third‑party dependencies.  Make sure to
clone the repository recursively:

```bash
git clone --recursive https://github.com/AppledoreM/CKMPM 
```

If you already cloned the repository without `--recursive`, initialise the
submodules manually:

```bash
git submodule update --init --recursive
```

## Building

Create a build directory and invoke CMake.  All targets can then be compiled via
`cmake --build` or your favourite build tool:

```bash
mkdir build
cd build
cmake ..
cmake --build . -j8
```

This will produce several test executables under `build/tests/`.  Each executable
is named `mpm_test_<name>` or similar, depending on the test case defined in
`tests/CMakeLists.txt`.

## Running the Tests

Run the desired test executable from the build directory.  For example:

```bash
./tests/mpm_test_dragon
```

Before running any tests, create a `result` directory in the repository
root to store output files:

```bash
mkdir -p result
```

Some tests export their results to `result/<test_name>/`. Tests are executed individually.

## Python Implementation

This repo also contains a python implementation of PIC version of CK-MPM, which is included in the `python-src` folder. Check out the `.sh` file on how to run the script.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@article{journals/corr/abs-2412-10399,
  added-at = {2025-01-20T00:00:00.000+0100},
  author = {Liu, Michael and Wang, Xinlei and Li, Minchen},
  biburl = {https://www.bibsonomy.org/bibtex/21a65ca05636892d8826138669f87c54d/dblp},
  ee = {https://doi.org/10.48550/arXiv.2412.10399},
  interhash = {b86884a3967d8fd38da875899a5c8e57},
  intrahash = {1a65ca05636892d8826138669f87c54d},
  journal = {CoRR},
  keywords = {dblp},
  timestamp = {2025-01-27T08:25:32.000+0100},
  title = {CK-MPM: A Compact-Kernel Material Point Method.},
  url = {http://dblp.uni-trier.de/db/journals/corr/corr2412.html#abs-2412-10399},
  volume = {abs/2412.10399},
  year = 2024
}


