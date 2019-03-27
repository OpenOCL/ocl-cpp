# ocl-cpp

## C++ code compilation

You can create a bash script `setup.sh` which will be ignored by git for your personal configuration, with the content along the lines of:
```bash
export CXX=g++
export LD_LIBRARY_PATH="/path/to/casadi/"
export DYLD_LIBRARY_PATH="/path/to/casadi/" # for mac os

export CASADI_INCLUDE="/path/to/casadi/include"
export CASADI_EXPORT="/path/to/casadi/build/"
export CASADI_LIB="/path/to/casadi/"
export GTEST_INCLUDE="/path/to/gtest"
```
where `INCLUDE` are the directories with the header files (`.h`) and LIB are the directories containing shared/static library files (`.a`, `.so`, `.dylib`).

For example:
```bash
export CXX=g++
export LD_LIBRARY_PATH=/home/jonas/casadi/build/lib
export DYLD_LIBRARY_PATH=/home/jonas/casadi/build/lib # for mac os

export CASADI_INCLUDE="/home/jonas/casadi"
export CASADI_EXPORT="/home/jonas/casadi/build/"
export CASADI_LIB="/home/jonas/casadi/build/lib"
export GTEST_INCLUDE="../googletest/googletest/"

```
and setup the environment variables with
```bash
source setup.sh
```

compile with
```bash
make
```

## Running tests

```bash
./build/bin/main_test
```

## Debugging

```bash
gdb ./build/bin/main_test
/Applications/Xcode.app/Contents/Developer/usr/bin/lldb ./build/bin/main_test # for mac os
```

## gdb commands

```bash
(b) break test_tree.cc:13
(r) run
(c) cont
(d) del 1
dis 1
en 1
(i) info break
(p) print x
```
