#include <utils/testing.h>
#include "tensor/root.h"


TEST(testRoot, Constructor) {

  auto branches = ocl::Root::Branches();
  IndizesArray indizes {{1,2,3,4,5,6},{7,8,9,10,11}};
  auto a = ocl::Root(branches, [2,3], indizes)
}
