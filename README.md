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
git clone --recursive https://github.com/HendrikMuenster/flexBox   #or "cd flexBox" and "git submodule update --init --recursive" if you cloned flexBox w/o CPP already
cd flexBox_CPP/source
mkdir build
cd build
cmake -D <variable1>=<value1> ../  #modify as desired see table below
make
make install                      #recommended but optional
```
The install target copies the compiled MEX-file and executables to build/bin where Matlab *should* find it.
Check the variable `params.relativePathToMex` if it doesn't and set accordingly. 

| CMake Variable  | Default Value   | Description                     |
| --------------- | :-------------: | ------------------------------- |
| BUILD_MATLAB    | ON              | compiles MEX-interface if ON    |
| BUILD_EXAMPLES  | ON              | compiles C++ stand-alone if ON  |
| USE_OPENMP      | OFF             | enables OpenMP support if ON    |
| USE_CUDA        | OFF             | enables CUDA support if ON      |

## Known Problems
### Linux
#### 'GLIBCXX' not found
The MEX-Interface is built with the Compiler found by CMake and thus could be unsupported by MathWorks (see [Supported Compilers](https://de.mathworks.com/support/sysreq/previous_releases.html)).
This could result in an issue similiar to <https://github.com/HendrikMuenster/flexBox_CPP/issues/2>. If this happens, you could either set a supported compiler in CMake via `export C=/usr/bin/supportedCompilerC` and `export CXX=/usr/bin/supportedCompilerCXX`,
or try to preload the correct libraries ***before*** starting Matlab like `LD_PRELOAD=/usr/bin/lib/libstdc++.so.6 matlab` (see [Matlab Answers](https://de.mathworks.com/matlabcentral/answers/329796-issue-with-libstdc-so-6)).
The real path and correct name depends on your specific environment. The error message should give you a hint which library is missing and the concrete library should be under `/usr/lib`, `/usr/lib32` or `/usr/lib64`. 
If you still can't make the MEX-Interface work, feel free to add another issue at <https://github.com/HendrikMuenster/flexBox_CPP/issues/2>

## Usage
We recommend to look at the provided examples in the folder examples/.
In the examples we use the [CImg Library](https://http://cimg.eu/) to process images.

The Doxygen documentation for the C++ module is available under the following link:
https://hendrikmuenster.github.io/flexBox_CPP/


## Reporting Bugs
In case you experience any problems, please create an issue at
https://github.com/HendrikMuenster/flexBox/issues or https://github.com/HendrikMuenster/flexBox_CPP/issues

