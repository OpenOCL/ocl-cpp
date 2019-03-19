# ocl-cpp

## C++ code compilation

You can create a bash script `compile.sh` which will be ignored by git for your personal configuration, with the content along the lines of:
```bash
set -e
export CXX=g++-4.8
export LD_LIBRARY_PATH=/path/to/casadi/

export CASADI_INCLUDE=/path/to/casadi/include
export CASADI_LIB=/path/to/casadi/
export GTEST_INCLUDE=/path/to/gtest
make gtest
make all -f Makefile
```
where `INCLUDE` are the directories with the header files (`.h`) and LIB are the directories containing shared/static library files (`.a`, `.so`, `.dylib`).

For example:
```bash
#!/bin/bash
set -e
export CXX=g++-4.8
export LD_LIBRARY_PATH=../../software/casadi-linux-matlabR2014b-v3.3.0/

export CASADI_INCLUDE="../../software/casadi-linux-matlabR2014b-v3.3.0/include/"
export CASADI_LIB="../../software/casadi-linux-matlabR2014b-v3.3.0/"
export GTEST_INCLUDE="../googletest/googletest/"
make gtest
make all -f Makefile
```
and run the compilation with
```bash
sh compile.sh
```

## Running
```
set -e
export DYLD_LIBRARY_PATH=../../Software/casadi-osx-matlabR2014b-v3.3.0/
./build/bin/test_tree
```

## Debugging
```
set -e
export DYLD_LIBRARY_PATH=../../Software/casadi-osx-matlabR2014b-v3.3.0/
/Applications/Xcode.app/Contents/Developer/usr/bin/lldb ./build/bin/test_tree

## gdb commands

```bash
b test_tree.cc:13
run
cont
del 1
dis 1
en 1
info break
print x
```


