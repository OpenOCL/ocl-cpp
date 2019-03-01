# SYNOPSIS:
#
#   make [all]  - makes everything.
#   make TARGET - makes the given target.
#   make clean  - removes all files generated by make.

CASADI = ../../Software/casadi-osx-matlabR2014b-v3.4.5/include/

GTEST = ../googletest/googletest
GTEST_LIB = $(GTEST)/lib

SRC = ./src
INCLUDE = ./include
TEST = ./test
LIB = ./lib
EXTERN = ./extern

BIN = ./build/bin
OBJ = ./build/obj

# Set Google Test's header directory as a system directory, such that
# the compiler doesn't generate warnings in Google Test headers.
CPPFLAGS += -isystem $(GTEST)/include -isystem $(CASADI)
CXXFLAGS += -g -Wall -Wextra -std=c++14 -Wno-ignored-attributes -Wfatal-errors

INCLUDES_EIGEN = -I $(EXTERN)/eigen
INCLUDES = -I$(SRC) -I$(INCLUDE)  $(INCLUDES_EIGEN) -I$(CASADI)

GTEST_STATIC = $(GTEST_LIB)/libgtest.a

TESTS =

GTEST_HEADERS = $(GTEST)/include/gtest/*.h \
                $(GTEST)/include/gtest/internal/*.h

all : $(GTEST_LIBS) $(TESTS) $(OBJ)/testTensor.o
gtest: $(GTEST_LIBS)
clean:
	rm -f $(TESTS) $(OBJ)/*.o	$(LIB)/*.a
clean-all :
	rm -f $(GTEST_LIBS) $(TESTS) $(OBJ)/*.o $(LIB)/*.a


# Builds gtest.a and gtest_main.a.
GTEST_SRCS_ = $(GTEST)/src/*.cc $(GTEST)/src/*.h $(GTEST_HEADERS)
$(OBJ)/gtest-all.o : $(GTEST_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST) $(CXXFLAGS) -c \
            $(GTEST)/src/gtest-all.cc -o $@
$(OBJ)/gtest_main.o : $(GTEST_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST) $(CXXFLAGS) -c \
            $(GTEST)/src/gtest_main.cc -o $@
$(GTEST_LIB)/libgtest.a : $(OBJ)/gtest-all.o
	$(AR) $(ARFLAGS) $@ $^
$(GTEST_LIB)/libgtest_main.a : $(OBJ)/gtest-all.o $(OBJ)/gtest_main.o
	$(AR) $(ARFLAGS) $@ $^

# builds Eigen Tensor
$(OBJ)/testTensor.o : $(TEST)/testTensor.cc $(SRC)/Tensor.h $(SRC)/NumericMatrix.cc $(SRC)/NumericMatrix.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
