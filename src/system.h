/*
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
#ifndef OCL_SYSTEM_H_
#define OCL_SYSTEM_H_

namespace ocl {

typedef SystemHandler SH;
typedef ImplicitEquationsHandler IEH;
typedef TreeTensor TT;



struct Bound
{
  Bound(const float_p lower_bound, const float_p upper_bound)
      : lower_bound(lower_bound), upper_bound(upper_bound) { }
  float_p lower_bound;
  float_p upper_bound;
};

class SystemVariablesHandler {

public:
  void state( const std::string& id, const std::vector<int>& shape = {1,1},
              const double& lower_bound = -std::numeric_limits<double>::infinity(),
              const double& upper_bound = std::numeric_limits<double>::infinity())
  {
    states_struct.add(id, shape);
    bounds[id] = Bound(lower_bound,upper_bound);
  }

  void algebraic( const std::string& id, const std::vector<int>& shape = {1,1},
                  const double& lower_bound = -std::numeric_limits<double>::infinity(),
                  const double& upper_bound = std::numeric_limits<double>::infinity())
  {
    algebraics_struct.add(id, shape);
    bounds[id] = Bound(lower_bound,upper_bound);
  }

  void control( const std::string& id, const std::vector<int>& shape = {1,1},
                const double& lower_bound = -std::numeric_limits<double>::infinity(),
                const double& upper_bound = std::numeric_limits<double>::infinity())
  {
    controls_struct.add(id, shape);
    bounds[id] = Bound(lower_bound,upper_bound);
  }

  void parameter( const std::string& id, const std::vector<int>& shape = {1,1},
                  const double& lower_bound = -std::numeric_limits<double>::infinity(),
                  const double& upper_bound = std::numeric_limits<double>::infinity())
  {
    parameters_struct.add(id, shape);
    bounds[id] = Bound(lower_bound,upper_bound);
  }

  Tree getStates() { return states_struct; }
  Tree getAlgebraics() { return algebraics_struct; }
  Tree getControls() { return controls_struct; }
  Tree getParameters() { return parameters_struct; }

private:
  std::map<std::string, Bound> bounds;
  Tree states_struct;
  Tree algebraics_struct;
  Tree controls_struct;
  Tree parameters_struct;
}

struct DifferentialEquation {
  void insert(const std::string& id, const TreeTensor& el)
  {
    std::pair<std::string, TreeTensor> pair(id, el);
    eq.insert(pair);
  }
  std::map<std::string, TreeTensor> eq;
}

struct ImplicitEquation {
  void insert(const TreeTensor& el) {
    eq.push_back(eq);
  }
  std::vector<TreeTensor> eq;
}

struct SystemEquation
{
  DifferentialEquation differential;
  ImplicitEquation implicit;
}

class SystemEquationsHandler
{
public:
  // sh.diff("x") = 2*u + x
  Equation& diff(const std::string& id)
  {
  }

  void differentialEquation(const std::string& id, const TreeTensor& eq) {
    eq.differential.insert(id, eq);
  }

  void implicitEquation(const TreeTensor& eq) {
    eq.implicit.insert(eq);
  }

  SystemEquation equation()
  {
    return eq;
  }

private:
  SystemEquation eq;
}

class System
{
public:
  System(const FunctionHandle variables, const FunctionHandle equations,
         const FunctionHandle initial_conditions
  {

    SystemVariablesHandler svh;
    variables.eval(svh);

    this->states_struct = svh.getStates();
    this->algvars_struct = svh.getAlgebraics();
    this->controls_struct = svh.getControls();
    this->parameters_struct = svh.getParameters();

    int sx = this->states_struct.size();
    int sz = this->algvars_struct.size();
    int su = this->controls_struct.size();
    int sp = this->parameters_struct.size();

    equations_fcn = Function(equations, {sx,sz,su,sp}, 2);
    ic_fcn = Function(initial_conditions, {sx,sp}, 1);
  }

  SystemEquation evaluate(const std::vector<float_p>& states, const std::vector<float_p>& algvars,
                          const std::vector<float_p>& controls, const std::vector<float_p>& parameters)
  {
    ValueStorage xvs(states);
    ValueStorage zvs(algvars);
    ValueStorage uvs(controls);
    ValueStorage pvs(parameters);

    SystemEquationsHandler eh;
    TT x = TreeTensor(states_struct, xvs);
    TT z = TreeTensor(algvars_struct, zvs);
    TT u = TreeTensor(controls_struct, uvs);
    TT p = TreeTensor(parameters_struct, pvs);

    equations_fcn.eval(eh, x, z, u, p);

    return eh.equations();
  }

private:
  Tree states_struct;
  Tree algvars_struct;
  Tree controls_struct;
  Tree parameters_struct;

  Function equations_fcn;
  Function ic_fcn;

} // namespace ocl
#endif // OCL_SYSTEM_H_
