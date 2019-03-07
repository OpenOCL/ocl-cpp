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
#ifndef OCLCPP_OCL_STRUCTUREDTENSOR_H_
#define OCLCPP_OCL_STRUCTUREDTENSOR_H_

#include "typedefs.h"

namespace ocl
{

class TreeTensor
{

 private:
  const ValueStorage &value;
  const Structure &structure;

 public:

  // Static factory methods
  static TreeTensor create(const Structure& structure, const ValueStorage& value);
  static TreeTensor create(const Structure& structure, const Tensor& value);
  static TreeTensor Matrix(const Tensor& value);

  // Constructor
  StructuredTensor(const Structure &structure, const ValueStorage &vs);
  // Returns the number of elements of the value
  int numel();
  // Returns the underlying value
  Tensor value();
  // Return a string representation
  std::string str();
  // Display
  void disp();
  // Sets a value, supports broadcasting
  void set(Tensor value);
  // Returns the size of value
  Size size();
  // Returns a sub-value by id
  Tensor get(std::string id);
  // Slices value
  Tensor slice(const Slice& slice1=Slice::all, const Slice& slice2=Slice::all,
               const Slice& slice3=Slice::all);

  // linspace operator
  T = linspace(const Value<T>& other);

  // operators - unary
  T uplus() return Value::Matrix(value);
  T uminus() return Value::Matrix(-value);
  T ctranspose() return this->transpose();
  T transpose() return Value::Matrix(value->transpose());
  T triu();
  T sum();
  T norm();
  T det();
  T trace();
  T diag();
  T abs();
  T sqrt();
  T sin();
  T cos();
  T tan();
  T atan();
  T asin();
  T acos();
  T tanh();
  T cosh();
  T sinh();
  T acosh();
  T exp();
  T log();
  // operators - unary + constant
  T reshape(int[]);
  T repmat(int[]);

}; // class StructuredTensor

} // namespace ocl


#endif  // OCLCPP_OCL_STRUCTUREDTENSOR_H_
