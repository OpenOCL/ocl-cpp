# SYNOPSIS:
#
#   make [all]  - makes everything.
#   make clean  - removes files generated by make.
#   make clean-all  - removes all files generated by make, including gtest and binaries.
#
# Be sure to set the following PATH variables:
#
# export CASADI_INCLUDE=/path/to/casadi/include
# export CASADI_LIB=/path/to/casadi/
# export GTEST_INCLUDE=/path/to/gtest

CASADI_INCLUDE_PATH = ${CASADI_INCLUDE}
CASADI_EXPORT_PATH = ${CASADI_EXPORT}
CASADI_LIB_PATH = ${CASADI_LIB}

GTEST_PATH = $(GTEST_INCLUDE)
GTEST_LIB = ./build/lib

SRC = ./src
INCLUDE = ./include
TEST = ./test
LIB = ./build/lib
EXTERN = ./extern

BIN = ./build/bin
OBJ = ./build/obj

CPPFLAGS += -isystem $(GTEST_PATH)\include -isystem $(CASADI_INCLUDE_PATH)
CXXFLAGS += -g -O0 -Wall -Wextra -std=c++11 -fPIC -Wno-delete-non-virtual-dtor

LDFLAGS +=

GTEST_LIBS = $(GTEST_LIB)/libgtest.a $(GTEST_LIB)/libgtest_main.a

INCLUDES_EIGEN = -I $(EXTERN)/eigen
INCLUDES = -I$(SRC) -I$(TEST) -I$(INCLUDE) \
					 $(INCLUDES_EIGEN) -I$(CASADI_INCLUDE_PATH) -I$(CASADI_EXPORT_PATH)

GTEST_STATIC = $(GTEST_LIB)/libgtest.a
GTEST_HEADERS = $(GTEST_PATH)/include/gtest/*.h \
                $(GTEST_PATH)/include/gtest/internal/*.h
GTEST_SRCS_ = $(GTEST_PATH)/src/*.cc $(GTEST_PATH)/src/*.h $(GTEST_HEADERS)

TEST_HEADERS = $(TEST)/test_casadi.h $(TEST)/test_matrix.h $(TEST)/test_tensor.h \
               $(TEST)/test_tree.h $(TEST)/test_tree_tensor.h $(TEST)/test_sym_matrix.h \
							 $(TEST)/test_system.h
COMMON_HEADERS = $(SRC)/utils/exceptions.h $(SRC)/utils/typedefs.h $(SRC)/utils/testing.h $(SRC)/utils/slicing.h $(SRC)/utils/assertions.h
TENSOR_HEADERS = $(SRC)/tensor/casadi.h $(SRC)/tensor/functions.h \
 					       $(SRC)/tensor/matrix.h  $(SRC)/tensor/tree.h \
								 $(SRC)/tensor/tensor.h $(SRC)/tensor/tree_builder.h \
								 $(SRC)/tensor/tree_tensor.h $(SRC)/tensor/value_storage.h
CORE_HEADERS = $(SRC)/function_interface.h $(SRC)/system.h

all: $(BIN)/main_test
playbox: $(BIN)/tensor_playbox
gtest: $(GTEST_LIBS)
clean:
	rm -f $(TESTS) $(OBJ)/*.o
clean-bin :
	rm -f build/bin/test_tensor
clean-all : clean-bin
	rm -f $(GTEST_LIBS) $(TESTS) $(OBJ)/*.o $(LIB)/*.a

# Rules for gtest.a and gtest_main.a.
$(OBJ)/gtest-all.o : $(GTEST_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST_PATH) $(CXXFLAGS) -c \
            $(GTEST_PATH)/src/gtest-all.cc -o $@
$(OBJ)/gtest_main.o : $(GTEST_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST_PATH) $(CXXFLAGS) -c \
            $(GTEST_PATH)/src/gtest_main.cc -o $@
$(GTEST_LIB)/libgtest.a : $(OBJ)/gtest-all.o
	$(AR) $(ARFLAGS) $@ $^
$(GTEST_LIB)/libgtest_main.a : $(OBJ)/gtest-all.o $(OBJ)/gtest_main.o
	$(AR) $(ARFLAGS) $@ $^

# playbox

$(OBJ)/tensor_playbox.o : $(TEST)/tensor_playbox.cc $(SRC)/tensor/matrix.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BIN)/tensor_playbox : $(OBJ)/tensor_playbox.o
	$(CXX) $(LDFLAGS) -L$(CASADI_LIB_PATH) $^ -lcasadi -o $@

#  compiles main test program
$(OBJ)/main_test.o : $(TEST)/main_test.cc $(TEST_HEADERS) $(TENSOR_HEADERS) $(COMMON_HEADERS) $(GTEST_HEADERS) $(CORE_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# links test program
$(BIN)/main_test : $(OBJ)/main_test.o
	$(CXX) $(LDFLAGS) -L$(GTEST_LIB) -L$(CASADI_LIB_PATH) $^ -lcasadi -lgtest_main -lpthread -o $@
