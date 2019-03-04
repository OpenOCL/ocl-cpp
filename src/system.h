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
#ifndef OCL_SYSTEM_H_
#define OCL_SYSTEM_H_

namespace ocl {

class NumericVector {

}

class SystemBuilder {

public:
  void addState(const std::string& id, const Shape& shape = {1,1},
                const Scalar& lower_bound = std::numeric_limits<double>::infinity(),
                const Scalar& upper_bound = -std::numeric_limits<double>::infinity())
  {
    states_struct.add(id, shape);
    bounds[id] = Bound(lower_bound,upper_bound);
  }

  void addAlgvar(const std::string& id, const Shape& shape = {1,1},
                 const Scalar& lower_bound = std::numeric_limits<double>::infinity(),
                 const Scalar& upper_bound = -std::numeric_limits<double>::infinity())
  {
    algvars_struct.add(id, shape);
    bounds[id] = Bound(lower_bound,upper_bound);
  }

  void addControl(const std::string& id, const Shape& shape = {1,1},
                  const Scalar& lower_bound = std::numeric_limits<double>::infinity(),
                  const Scalar& upper_bound = -std::numeric_limits<double>::infinity())
  {
    controls_struct.add(id, shape);
    bounds[id] = Bound(lower_bound,upper_bound);
  }

  void addParameter(const std::string& id, const Shape& shape = {1,1},
                    const Scalar& lower_bound = std::numeric_limits<double>::infinity(),
                    const Scalar& upper_bound = -std::numeric_limits<double>::infinity())
  {
    parameters_struct.add(id, shape);
    bounds[id] = Bound(lower_bound,upper_bound);
  }

private:
  std::map<std::string, Bound> bounds;
  RootNode states_struct;
  RootNode algvars_struct;
  RootNode controls_struct;
  RootNode parameters_struct;
}

class ImplicitEquationsHandler
{

}

class ImplicitEquations
{

}

class System
{
public:
  System(const FunctionHandle variables, const FunctionHandle equations,
         const FunctionHandle initial_conditions,
         const RootNode& states_struct, const RootNode& algvars_struct,
         const RootNode& controls_struct, const RootNode& parameters_struct)
      : states_struct(states_struct), algvars_struct(algvars_struct),
        controls_struct(controls_struct), parameters_struct(parameters_struct)

    SystemBuilder sb;
    variables.eval(sb);

    states_struct = sb.getStatesStruct();
    algvars_struct = sb.getAlgvarsStruct();
    controls_struct = sb.getControlsStruct();
    parameters_struct = sb.getParametersStruct();

    int sx = statesStruct.size();
    int sz = algVarsStruct.size();
    int su = controlsStruct.size();
    int sp = parametersStruct.size();

    equations_fcn = Function(equations, {sx,sz,su,sp}, 2);
    ic_fcn = Function(initial_conditions, {sx,sp}, 1);
  }

  void setup();
  ImplicitEquations evalEquations(const NumericVector& states, const NumericVector& algvars,
                                 const NumericVector& controls, const NumericVector& parameters)
  {
    ImplicitEquationsHandler eh;
    x = Tensor.create(states_struct, states);
    z = Tensor.create(algvars_struct, algvars);
    u = Tensor.create(controls_struct, controls);
    p = Tensor.create(parameters_struct, parameters);
    equations_fcn.eval(eh, x, z, u, p);
    return eh.equations();
  }

  ImplicitEquation evalInitialConditions(const NumericVector& states, const NumericVector& parameters)
  {
    ImplicitEquationsHandler eh;
    x = Tensor.create(states_struct, states);
    p = Tensor.create(parameters_struct, parameters);
    equations_fcn.eval(eh, x, p);
    return eh.equations();
  }

private:
  RootNode states_struct;
  RootNode algvars_struct;
  RootNode controls_struct;
  RootNode parameters_struct;

  Function equations_fcn;
  Function ic_fcn;

} // namespace ocl
#endif // OCL_SYSTEM_H_
