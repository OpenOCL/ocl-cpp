#include <iostream>
#include "tensor/casadi.h"

int main()
{
  ::casadi::DM x = ::casadi::DM::zeros(3,3);

  x.set(3, false, 0, 2); 

  std::cout << x << std::endl;

}
