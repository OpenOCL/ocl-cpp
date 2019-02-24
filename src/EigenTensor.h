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

#include <unsupported/Eigen/CXX11/Tensor>

#include "typedefs.h"



namespace ocl
{

template <typename N>
class EigenTensor : public Tensor
{

 private:
  eigen::Tensor<float, N> tensor;

 public:

  // Constructor
  EigenTensor(eigen::Tensor tensor) : tensor(tensor) { }

  // Returns the underlying value
  eigen::Tensor value();
  // Return a string representation
  std::string str();
  // Sets a value, supports broadcasting
  void set(eigen::Tensor value);
  // Slices value
  Tensor slice(String slice1=":", String slice2=":", String slice3=":");

  // linspace operator
  Tensor linspace(const Tensor& other);

  // operators - unary element wise

  Tensor uplus() { return EigenTensor(tensor); }
  Tensor uminus() { return EigenTensor(-tensor); }
  Tensor square() { return EigenTensor(tensor.square()); }
  Tensor inv() { return EigenTensor(tensor.inverse()); }
  Tensor ctranspose() { return EigenTensor(tensor.transpose()); }
  Tensor transpose() { return EigenTensor(tensor.transpose()); }
  T triu();
  T sum();
  T norm();
  T det();
  T trace();
  T diag();
  Tensor abs() { return EigenTensor(tensor.abs()); }
  Tensor sqrt() { return EigenTensor(tensor.sqrt()); }
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
  T exp() { return EigenTensor(tensor.exp()); }
  T log() { return EigenTensor(tensor.log()); }

  // operators - unary element wise + constant
  Tensor pow(double exponent)  { return EigenTensor(tensor.pow(exponent)); }
  T reshape(int[]);
  T repmat(int[]);

  // operators - binary element wise

}; // class Value

} // namespace ocl


#endif  // OCLCPP_OCL_NUMERICVALUE_H_
