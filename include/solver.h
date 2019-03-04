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
#ifndef OCL_SOLVER_H_
#define OCL_SOLVER_H_

namespace ocl {


struct FunctionHandle
{
  const void* fh;
}

struct Options
{
  // general
  std::string solver_interface         = "casadi";
  bool system_casadi_mx                = false;
  bool controls_regularization         = true;
  double controls_regularization_value = 1e-6;
  bool path_constraints_at_boundary    = true;

  // nlp
  std::string discretization  = "collocation";
  int control_intervals       = 20;
  int collocation_order       = 3;
  std::string solver          = "ipopt";
  bool auto_interpolation     = true;

  // casadi options

  // ipopt options

}

struct Bound
{
  double lower;
  double upper;
}

struct OcpDefinition
{
  // mandatory
  FunctionHandle variables;
  FunctionHandle equations;

  // optional
  FunctionHandle path_costs;
  FunctionHandle arrival_costs;
  FunctionHandle path_constraints;
  FunctionHandle boundary_conditions;

  // optional discrete
  FunctionHandle discrete_costs;
  FunctionHandle discrete_constraints;
}

class Solver
{
public:
  Solver(const FunctionHandle variables, const FunctionHandle equations,
         const FunctionHandle path_costs, const FunctionHandle arrival_costs,
         const FunctionHandle path_constraints,
         const FunctionHandle boundary_conditions)
  {
    OcpDefinition ocp;
    ocp.variables = variables;
    ocp.equations = equations;
    ocp.path_costs = path_costs;
    ocp.arrival_costs = arrival_costs;
    ocp.path_constraints = path_constraints;
    ocp.boundary_conditions = boundary_conditions;
    Solver(ocp);
  }

  Solver(OcpDefinition ocp {
    Solver({ocp});
  }

  Solver(const std::vector<OcpDefinition>& ocp) :
      options(), ocp(ocp) { }

  void setInitialBounds(const std::string& var, const Bound bound);
  void setEndBounds(const std::string& var, const Bound bound);
  void setBounds(const std::string& var, const Bound bound);

  Solution solve() { return solve(this->initialGuess); }
  Solution solve(const InitialGuess& initial_guess);

  Options options;
  InitialGuess initial_guess;
private:
  std::vector<OcpDefinition> ocp;
}

} // namespace ocl
#endif // OCL_SOLVER_H_
