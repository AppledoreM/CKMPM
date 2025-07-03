# Compact-Kernel (CK) MPM 
![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

This repository contains the official implementation of the [Compact-Kernel Material Point Method (CK-MPM)](https://arxiv.org/pdf/2412.10399) and a set of example tests. The tests are built as standalone executables using
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
@article{liu2025ckmpm,
  author = {Michael Liu, Xinlei Wang, Minchen Li},
  title = {CK-MPM: A Compact-Kernel Material Point Method},
  year = {2025},
  issue_date = {July 2025},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  volume = {44},
  number = {4},
  issn = {0730-0301},
  url = {https://doi.org/10.1145/3731155},
  doi = {10.1145/3731155},
  journal = {ACM Trans. Graph.},
  month = july,
  numpages = {14}
}
