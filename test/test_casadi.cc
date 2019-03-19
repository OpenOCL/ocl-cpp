#include <utils/testing.h>
#include "tensor/casadi.h"


TEST(testCasadiMatrix, ScalarOperators)
{
  // scalar unary operations
  {
    ocl::CasadiMatrix m = ocl::CasadiMatrix(4);
    ocl::test::assertEqual( ocl::casadi::full(m), {4} );
  }
}
