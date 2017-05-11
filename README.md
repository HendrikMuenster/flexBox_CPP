# FlexBox - A **Flex**ible Primal-Dual Tool**Box**

## Introduction
This repository contains the C++ module for **FlexBox**. Please refer to the main repository for licensing and citation.

The main repository can be found under
https://github.com/HendrikMuenster/flexBox

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
cd flexBox_CPP
mkdir build
cd build
cmake -D <variable1>=<value1> ../  #modify as desired see table below
make
make install                      #recommended but optional
```
The install target copies the compiled MEX-file and executables to build/bin.

| CMake Variable  | Default Value   | Description                     |
| --------------- | :-------------: | ------------------------------- |
| BUILD_MATLAB    | ON              | compiles MEX-interface if ON    |
| BUILD_EXAMPLES  | ON              | compiles C++ stand-alone if ON  |
| USE_OPENMP      | OFF             | enables OpenMP support if ON    |
| USE_CUDA        | OFF             | enables CUDA support if ON      |

## Usage
We recommend to look at the provided examples in the folder examples/.
In the examples we use the [CImg Library](https://http://cimg.eu/) to process images.

The Doxygen documentation for the C++ module is available under the following link:
https://hendrikmuenster.github.io/flexBox_CPP/


## Reporting Bugs
In case you experience any problems, please create an issue at
https://github.com/HendrikMuenster/flexBox/issues or https://github.com/HendrikMuenster/flexBox_CPP/issues
