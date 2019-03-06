# ocl-cpp

## C-code compilation

You can create a bash script `compile.sh` which will be ignored by git for your personal configuration, with the content along the lines of:
```bash
CASADI_INCLUDE=/path/to/casadi/include; \
CASADI_LIB=/path/to/casadi/; \
GTEST_INCLUDE=/path/to/gtest;\
make
```
where `INCLUDE` are the directories with the header files (`.h`) and LIB are the directories containing shared/static library files (`.a`, `.so`, `.dylib`).

and run the compilation with
```bash
sh compile.sh
```
