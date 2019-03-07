#include <test.h>

#include "casadi.h"


TEST(testCasadiMatrix, ScalarOperators)
{
  // scalar unary operations
  {
    CasadiMatrixNat m = CasadiMatrixNat(4);
    ocl::test::assertDoubleFullEqual( ocl::full(m), 4 );
  }
}
