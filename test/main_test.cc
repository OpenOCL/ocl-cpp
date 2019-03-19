#include "test_casadi.cc"
#include "test_matrix.cc"
#include "test_tree.cc"
#include "test_tensor.cc"

#include <gtest/gtest.h>

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
