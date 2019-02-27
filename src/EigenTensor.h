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
#ifndef OCLCPP_OCL_EIGENTENSOR_H_
#define OCLCPP_OCL_EIGENTENSOR_H_

#include <cmath>
#include <functional>

#include <unsupported/Eigen/CXX11/Tensor>

#include "typedefs.h"
#include "exceptions.h"


namespace ocl
{

template<int R>
class Tensor
{

 public:
  typedef Eigen::Tensor<float, R> EigenTensorRX;
  typedef typename EigenTensorRX::Dimensions Dimensions; // need to add keyword typename because it is a dependent type
  typedef float EigenScalar;
  typedef Eigen::array<Eigen::DenseIndex, R> StartIndices;
  typedef Eigen::array<Eigen::DenseIndex, R> Sizes;
  typedef Eigen::array<bool, R> ReverseDimensions;

  // static constructors
  Tensor FromTensorRX(const EigenTensorRX& tensor) {
    Tensor t = Tensor();
    t.tensor = tensor;
    return t;
  }

  // Constructor
  Tensor(const Sizes& dims) : tensor(EigenTensorRX(dims).setZero()) { }
  Tensor() { }

  // Returns the underlying value
  EigenTensorRX value();
  // Return a string representation
  std::string str();
  // Sets a value, supports broadcasting
  void set(const typename Eigen::internal::Initializer<EigenTensorRX, R>::InitList& values) { tensor.setValues(values); }
  // Slices value
  EigenTensorRX slice(String slice1=":", String slice2=":", String slice3=":");

  // linspace operator
  EigenTensorRX linspace(const Tensor& other);

  // operators - unary element wise
  Tensor uplus() { return Tensor::FromTensorRX(tensor); }
  Tensor uminus() { return Tensor::FromTensorRX(-tensor); }
  Tensor square() { return Tensor::FromTensorRX(tensor.square()); }
  Tensor inv() { return Tensor::FromTensorRX(tensor.inverse()); }
  Tensor abs() { return Tensor::FromTensorRX(tensor.abs()); }
  Tensor sqrt() { return Tensor::FromTensorRX(tensor.sqrt()); }
  Tensor sin() { return Tensor::FromTensorRX(tensor.unaryExpr(Eigen::internal::scalar_sin_op<EigenScalar>())); }
  Tensor cos() { return Tensor::FromTensorRX(tensor.unaryExpr(Eigen::internal::scalar_cos_op<EigenScalar>())); }
  Tensor tan() { return Tensor::FromTensorRX(tensor.unaryExpr(Eigen::internal::scalar_tan_op<EigenScalar>())); }
  Tensor atan() { return Tensor::FromTensorRX(tensor.unaryExpr(Eigen::internal::scalar_atan_op<EigenScalar>())); }
  Tensor asin() { return Tensor::FromTensorRX(tensor.unaryExpr(Eigen::internal::scalar_asin_op<EigenScalar>())); }
  Tensor acos() { return Tensor::FromTensorRX(tensor.unaryExpr(Eigen::internal::scalar_acos_op<EigenScalar>())); }
  Tensor tanh() { return Tensor::FromTensorRX(tensor.unaryExpr(Eigen::internal::scalar_tanh_op<EigenScalar>())); }
  Tensor cosh() { return Tensor::FromTensorRX(tensor.unaryExpr(Eigen::internal::scalar_cosh_op<EigenScalar>())); }
  Tensor sinh() { return Tensor::FromTensorRX(tensor.unaryExpr(Eigen::internal::scalar_sinh_op<EigenScalar>())); }
  Tensor acosh() { throw NotImplemented("No Eigen acosh support."); return Tensor({0,0,0}); }
  Tensor exp() { return Tensor::FromTensorRX(tensor.exp()); }
  Tensor log() { return Tensor::FromTensorRX(tensor.log()); }

  // operators - unary element wise + constant
  Tensor pow(double exponent)  { return Tensor::FromTensorRX(tensor.pow(exponent)); }

  // operators - reduction
  Tensor sum() { return Tensor::FromTensorRX(tensor.sum()); }
  Tensor sum(const Dimensions& dims) { return Tensor::FromTensorRX(tensor.sum(dims)); }
  Tensor max() { return Tensor::FromTensorRX(tensor.maximum()); }
  Tensor max(const Dimensions& dims) { return Tensor::FromTensorRX(tensor.maximum(dims)); }
  Tensor min() { return Tensor::FromTensorRX(tensor.minimum()); }
  Tensor min(const Dimensions& dims) { return Tensor::FromTensorRX(tensor.minimum(dims)); }
  Tensor prod() { return Tensor::FromTensorRX(tensor.prod()); }
  Tensor prod(const Dimensions& dims) { return Tensor::FromTensorRX(tensor.prod(dims)); }

  // operators - geometrical
  Tensor reshape(const Dimensions& dims) { return Tensor::FromTensorRX(tensor.reshape(dims)); }
  Tensor transpose(const Dimensions& dims) { return Tensor::FromTensorRX(tensor.shuffle(dims)); }
  Tensor slice(const StartIndices& offsets, const Sizes& extends) {
    return Tensor::FromTensorRX(tensor.slice(offsets,extends));
  }
  Tensor reverse(const ReverseDimensions& dims) {
    return Tensor::FromTensorRX(tensor.reverse(dims));
  }
  Tensor repeat(const Dimensions& dims) {
    return Tensor::FromTensorRX(tensor.broadcast(dims));
  }

  // operators - binary element wise
  Tensor operator+(const Tensor& other) {
    return Tensor::FromTensorRX(tensor+other.tensor);
  }
  Tensor operator-(const Tensor& other) {
    return Tensor::FromTensorRX(tensor-other.tensor);
  }
  Tensor operator*(const Tensor& other) {
    return Tensor::FromTensorRX(tensor*other.tensor);
  }
  Tensor operator/(const Tensor& other) {
    return Tensor::FromTensorRX(tensor/other.tensor);
  }

  // operators - binary

  // tensor multiplication (like matrix multiplication)
  //Tensor mul(const EigenTensor& other, const DimensionList& dims) {
  //  return EigenTensor(tensor.contract(other.tensor, dims));
  //}

  // operator - matrix
  // Tensor trace() { return EigenTensor(tensor.trace()); }
  // Tensor trace(const Dimensions& dims) { return EigenTensor(tensor.trace(dims)); }
  // Tensor diag();
  // Tensor triu();
  // Tensor norm();
  // Tensor det();

 private:
  EigenTensorRX tensor;

}; // class EigenTensor

} // namespace ocl


#endif  // OCLCPP_OCL_EIGENTENSOR_H_
