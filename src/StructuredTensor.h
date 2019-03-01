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

class StructuredTensor
{

 private:
  const ValueStorage &value;
  const Structure &structure;

 public:

  // Static factory methods
  static StructuredTensor create(const Structure &structure, const Tensor &value);
  static StructuredTensor create(const Structure &structure, const float[] &value);
  static StructuredTensor createFromValue(const Structure &structure,
      const PositionArray &positions, const ValueStorage &value);
  static StructuredTensor Matrix(const float[] &value);

  // Constructor
  StructuredTensor(const Structure &structure, const ValueStorage &vs);
  // Returns the number of elements of the value
  int numel();
  // Returns the underlying value
  T value();
  // Return a string representation
  std::string str();
  // Display
  void disp();
  // Sets a value, supports broadcasting, optional slices
  void set(T value, std::string slice1=":",
      std::string slice2=":", std::string slice3=":");
  // Returns the size of value
  Size size();
  // Returns a sub-value by id
  T get(std::string id);
  // Slices value
  T slice(std::string slice1=":",
      std::string slice2=":", std::string slice3=":");

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
