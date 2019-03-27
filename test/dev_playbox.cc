#include <iostream>
#include "tensor/casadi.h"
#include "tensor/tree_builder.h"

int main()
{
  ::casadi::DM x = ::casadi::DM::zeros(3,3);

  x = vertcat(x,x);
 
  std::cout << x << std::endl;


  x.set(3, false, 0, 2);

  ocl::TreeBuilder tb;
  tb.add("x1", {1,2});
  tb.add("x2", {3,2});

  std::cout << ocl::casadi::full(ocl::casadi::Eye(3)) << std::endl;

}
