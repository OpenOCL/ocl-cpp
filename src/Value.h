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
 *    You should have received a copy of the GNU General Public
 *    License along with this program;
 *    if not, write to the Free Software
 *    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */
#ifndef OCLCPP_OCL_NUMERICVALUE_H_
#define OCLCPP_OCL_NUMERICVALUE_H_

#include "typedefs.h"

namespace ocl
{

template<class T>
class Value
{

 private:
  const T &value;
  const Structure &structure;
  const PositionArray &positions;

 public:

  // Static factory methods
  static Value<T> create(const Structure &structure, const T &value);
  static Value<T> create(const Structure &structure, const float[] &value);
  static Value<T> createFromValue(const Structure &structure,
      const PositionArray &positions, const Value<T> &value);
  static Value<T> Matrix(const float[] &value);

  // Constructor
  Value(const Structure &structure, const PositionArray &positions, const T &value);
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

}; // class Value

} // namespace ocl


#endif  // OCLCPP_OCL_NUMERICVALUE_H_
