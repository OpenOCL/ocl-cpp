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

#include "utils/typedefs.h"
#include "tensor/tree_tensor.h"
#include "Function.h"

namespace ocl {

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

  Tree getStates() { return states_struct.tree(); }
  Tree getAlgebraics() { return algebraics_struct.tree(); }
  Tree getControls() { return controls_struct.tree(); }
  Tree getParameters() { return parameters_struct.tree(); }

private:
  std::map<std::string, Bound> bounds;
  TreeBuilder states_struct;
  TreeBuilder algebraics_struct;
  TreeBuilder controls_struct;
  TreeBuilder parameters_struct;
};

struct DifferentialEquation {
  void insert(const std::string& id, const TreeTensor& el)
  {
    std::pair<std::string, TreeTensor> pair(id, el);
    eq.insert(pair);
  }

  TreeTensor get(const std::string& id) {
    return eq.at(id);
  }

  std::map<std::string, TreeTensor> eq;
};

struct ImplicitEquation
{
  void append(const TreeTensor& el) {
    eq.push_back(el);
  }
  std::vector<TreeTensor> eq;
};

struct SystemEquation
{
  DifferentialEquation differential;
  ImplicitEquation implicit;
};

class SystemEquationsHandler
{
public:
  // sh.diff("x") << 2*u + x or sh.diff("x") = 2*u + x
  // TreeTensor& diff(const std::string& id)
  // {
  // }

  void differentialEquation(const std::string& id, const TreeTensor& ode) {
    sys_eq.differential.insert(id, ode);
  }

  void implicitEquation(const TreeTensor& alg) {
    sys_eq.implicit.append(alg);
  }

  SystemEquation equation()
  {
    return sys_eq;
  }

private:
  SystemEquation sys_eq;
};

typedef SystemVariablesHandler SVH;
typedef SystemEquationsHandler SEH;
typedef TreeTensor TT;

class System
{
public:

  typedef void (*VariablesFunctionPtr)(SystemVariablesHandler& sh);
  typedef void (*EquationsFunctionPtr)(SystemEquationsHandler& eh, const TreeTensor& x, const TreeTensor& z, const TreeTensor& u, const TreeTensor& p);



  System(const VariablesFunctionPtr variables_fcn_ptr, const EquationsFunctionPtr equations_fcn_ptr)
  {
    SystemVariablesHandler svh;
    variables_fcn_ptr(svh);

    this->states_struct = svh.getStates();
    this->algvars_struct = svh.getAlgebraics();
    this->controls_struct = svh.getControls();
    this->parameters_struct = svh.getParameters();

    int sx = this->states_struct.size();
    int sz = this->algvars_struct.size();
    int su = this->controls_struct.size();
    int sp = this->parameters_struct.size();


    void equations_fcn_ptr_remapped(SystemEquationsHandler& eh, const std::vector<TreeTensor>& args) {
      equations_fcn_ptr(eh, args[0], args[1], args[2], args[3]);
    }

    equations_fcn = Function(&equations_fcn_ptr_remapped, {sx,sz,su,sp}, 2);
  }

  SystemEquation evaluate(const Matrix& states, const Matrix& algvars,
                          const Matrix& controls, const Matrix& parameters)
  {
    ValueStorage x_vs(states);
    ValueStorage z_vs(algvars);
    ValueStorage u_vs(controls);
    ValueStorage p_vs(parameters);

    SystemEquationsHandler eh;
    TreeTensor x = TreeTensor(this->states_struct, x_vs);
    TreeTensor z = TreeTensor(this->algvars_struct, z_vs);
    TreeTensor u = TreeTensor(this->controls_struct, u_vs);
    TreeTensor p = TreeTensor(this->parameters_struct, p_vs);

    equations_fcn.eval(eh, {x, z, u, p});

    return eh.equations();
  }

private:
  Tree states_struct;
  Tree algvars_struct;
  Tree controls_struct;
  Tree parameters_struct;

  Function equations_fcn;
  Function ic_fcn;

};

} // namespace ocl
#endif // OCL_SYSTEM_H_
