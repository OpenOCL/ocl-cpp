#include <gtest/gtest.h>

#include "test_casadi.h"
#include "test_matrix.h"
#include "test_tree.h"
#include "test_tensor.h"
#include "test_tree_tensor.h"
#include "test_sym_matrix.h"

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
