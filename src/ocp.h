/*
 *
 *    Copyright (C) 2019 Jonas Koenemann
 *
 *    This program is free software; you can redistribute it and/or
 *    modify it under the terms of the GNU General Public
 *    License as published by the Free Software Foundation; either
 *    version 3 of the License, or (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    General Public License for more details.
 *
 */
#ifndef OCL_OCP_H_
#define OCL_OCP_H_

namespace ocl {

  FunctionHandle path_costs;
  FunctionHandle arrival_costs;
  FunctionHandle path_constraints;
  FunctionHandle boundary_conditions;


class OcpHandler
{

}

class OCP
{
public:
  OCP(const System& system,
      const FunctionHandle& path_costs,
      const FunctionHandle& arrival_costs,
      const FunctionHandle& path_constraints,
      const FunctionHandle& boundary_conditions)
  {

    int sx = system.sx();
    int sz = system.sz();
    int su = system.su();
    int sp = system.sp();

    path_cost_fcn = Function(path_costs, {sx,sz,su,sp}, 1);
    arrival_cost_fcn = Function(arrival_costs, {sx,sp}, 1);
    path_constraints_fcn = Function(path_constraints, {sx,sx,sp}, 3);
    boundary_conditions_fcn = Function(boundary_conditions, {sx,sp}, 3);
  }

  CostTerm evalPathCosts(const NumericVector& states, const NumericVector& algvars,
                         const NumericVector& controls, const NumericVector& parameters)
  {
    CostHandler ch;
    x = TreeTensor.create(states_struct, states);
    z = TreeTensor.create(algvars_struct, algvars);
    u = TreeTensor.create(controls_struct, controls);
    p = TreeTensor.create(parameters_struct, parameters);
    path_cost_fcn.eval(ch, x, z, u, p);
    return ch.costs();
  }

  CostTerm evalArrivalCosts(const NumericVector& states, const NumericVector& parameters)
  {
    CostHandler ch;
    x = TreeTensor.create(states_struct, states);
    p = TreeTensor.create(parameters_struct, parameters);
    path_cost_fcn.eval(ch, x, p);
    return ch.costs();
  }

  CostTerm evalPathConstraints(const NumericVector& states, const NumericVector& parameters)
  {
    CostHandler ch;
    x = TreeTensor.create(states_struct, states);
    p = TreeTensor.create(parameters_struct, parameters);
    path_constraints_fcn.eval(ch, x, p);
    return ch.costs();
  }

  CostTerm evalPathConstraints(const NumericVector& states0, const NumericVector& statesF,
                               const NumericVector& parameters)
  {
    CostHandler ch;
    x0 = TreeTensor.create(states_struct, states0);
    xF = TreeTensor.create(states_struct, statesF);
    p = TreeTensor.create(parameters_struct, parameters);
    boundary_conditions_fcn.eval(ch, x0, xF, p);
    return ch.costs();
  }

private:
  Function path_cost_fcn;
  Function arrival_cost_fcn;
  Function path_constraints_fcn;
  Function boundary_conditions_fcn;
}

} // namespace ocl
#endif // OCL_OCP_H_
