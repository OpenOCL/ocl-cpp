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
#ifndef OCL_FUNCTION_H_
#define OCL_FUNCTION_H_

namespace ocl {

class FunctionInterface
{
public:
  virtual std::vector<Matrix> fcnEvaluate(const std::vector<Matrix>& args) const = 0;
};

class Function
{
public:

  Function() {}
  Function(FunctionInterface *input_obj_ptr, const std::vector<int>& inputs, const int n_outputs) : obj_ptr(input_obj_ptr)
  {
    // does nothing, disables warning of unused input
    (void)inputs;
    (void)n_outputs;
  }

  std::vector<Matrix> evaluate(const std::vector<Matrix>& args)
  {
    return this->obj_ptr->fcnEvaluate(args);
  }

private:
  FunctionInterface *obj_ptr;
};

} // namespace ocl
#endif // OCL_FUNCTION_H_
