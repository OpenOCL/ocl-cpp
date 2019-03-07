#include <utils/test.h>

#include "tensor/casadi.h"


TEST(testCasadiMatrix, ScalarOperators)
{
  // scalar unary operations
  {
    ocl::CasadiMatrixNat m = ocl::CasadiMatrixNat(4);
    ocl::test::assertEqual( ocl::casadi::full(m), {4} );
  }
}
