# FlexBox - A **Flex**ible Primal-Dual Tool**Box**

## Introduction
This repository contains the C++ module for **FlexBox**. Please refer to the main repository for licensing and citation.

The main repository can be found under
https://github.com/HendrikMuenster/flexBox_CPP.

## Authors
* Hendrik Dirks ([hendrik.dirks@wwu.de](mailto:hendrik.dirks@wwu.de))*
* Lars Haalck ([lars.haalck@wwu.de](mailto:lars.haalck@wwu.de))*

\*Institute for Computational and Applied Mathematics
University of Muenster, Germany

## License
FlexBox is copyright ©2016-2017 by Hendrik Dirks.
If you plan to distribute the software (commercially or not), please contact Hendrik Dirks for more information.

## Installation

### Dependencies
In order to use the MEX interface the following requirements should be met:
* Matlab >= R2015b
* Image Processing Toolbox
* CMake >= 3.0.2
* OpenMP >= 2.0 when building with OpenMP support (optional)
* CUDA >= 7.5 when building with CUDA support (optional)

In order to use the stand-alone version the following requirements should be met:
* CMake >= 3.0.2
* OpenMP >= 2.0 when building with OpenMP support (optional)
* CUDA >= 7.5 when building with CUDA support (optional)

### Quick start
```
cd flexBox_CPP/
cmake ../
make
make install
```

## FAQ
You may have to preload libraries. To do so, simply start Matlab with:
env LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6 /usr/local/cuda-7.5/lib64/libcudart.so.7.5 /usr/local/cuda-7.5/lib64/libcusparse.so.7.5" matlab



## Reporting Bugs
In case you experience any problems, please create an issue at https://github.com/HendrikMuenster/flexBox_CPP/issues
