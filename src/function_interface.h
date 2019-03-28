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
#ifndef OCL_FUNCTION_INTERFACE_H_
#define OCL_FUNCTION_INTERFACE_H_

namespace ocl {

class FunctionInterface
{
public:

  FunctionInterface() {}
  FunctionInterface(const std::vector<Tree>& inputs, const int n_outputs)
  {
    // does nothing, disables warning of unused input
    (void)inputs;
    (void)n_outputs;
  }

  virtual std::vector<Matrix> fcnEvaluate(const std::vector<Matrix>& args) const = 0;

  std::vector<Matrix> evaluate(const std::vector<Matrix>& args)
  {
    return this->fcnEvaluate(args);
  }

private:

};

} // namespace ocl
#endif // OCL_FUNCTION_INTERFACE_H_
