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

double mysin(double x)
{
  return sin(x);
}



class TensorValue
{
public:
  virtual void data(void* vout) = 0;
}

class EigenTensorValue : public TensorValue
{
public: 
  virtual void data(void* vout)
  {

  }
private:
  Eigen::Operator<> v;
}



class TensorOperator
{
public:
  virtual TensorValue eval() = 0;
};

class TensorConstant : TensorOperator
{
public:
  double eval() override
}

class EigenTensorConstant : TensorConstant
{
public:
  EigenNativeOperator eval() override 
  {
    return Eigen::Tensor();
  }
}

class EigenTensorOperator : public TensorOperator
{
public:
  
};

class EigenUnaryOperator : public EigenTensorOperator
{
public:
  EigenUnaryOperator(TensorOperator input)
      : input(input.v) { }

private:
  TensorOperator input;
};

class EigenBinaryOperator : public EigenTensorOperator
{
public:
  EigenBinaryOperator(TensorOperator input1, TensorOperator input2)
      : input1(input1.v), input2(input2.v) { }

private:
  TensorOperator input1;
  TensorOperator input1;
};

class EigenUplus : public EigenUnaryOperator
{
  EigenUplus(TensorOperator input): EigenUnaryOperator(input) { }
  TensorValue eval() { return EigenTensorValue(input.eval().value()); }

  Eigen::Operator<> getOp1()
  {
    Eigen::Operator<> op1;
    input1.eval().data(op1);
    return op1;
  }
}

class EigenPlus : public EigenBinaryOperator
{
  EigenPlus(TensorOperator input1, TensorOperator input2): EigenBinaryOperator(input1, input2) { }
  TensorValue eval() overwrite { 

    Eigen::Operator<> op1 = getOp1();
    Eigen::Operator<> op2 = getOp2();
    return EigenTensorValue(op1 + op2);
  }
}




template<int R>
class EigenTensor : public Tensor
{

 public:
  typedef Eigen::Tensor<float, R> EigenTensorRX;
  typedef typename EigenTensorRX::Dimensions Dimensions; // need to add keyword typename because it is a dependent type
  typedef float EigenScalar;
  typedef Eigen::array<Eigen::DenseIndex, R> StartIndices;
  typedef Eigen::array<Eigen::DenseIndex, R> Sizes;
  typedef Eigen::array<bool, R> ReverseDimensions;

  // Constructor
  EigenTensor(EigenTensorRX tensor) : tensor(tensor) { }
  EigenTensor(Dimensions dims) : tensor(EigenTensorRX(dims).setZero()) { }

  // Returns the underlying value
  EigenTensorRX value();
  // Return a string representation
  std::string str();
  // Sets a value, supports broadcasting
  void set(EigenTensorRX value);
  // Slices value
  Tensor slice(String slice1=":", String slice2=":", String slice3=":");

  // linspace operator
  Tensor linspace(const Tensor& other);

  // operators - unary element wise
  Tensor uplus() { return EigenTensor(tensor); }
  Tensor uminus() { return EigenTensor(-tensor); }
  Tensor square() { return EigenTensor(tensor.square()); }
  Tensor inv() { return EigenTensor(tensor.inverse()); }
  Tensor abs() { return EigenTensor(tensor.abs()); }
  Tensor sqrt() { return EigenTensor(tensor.sqrt()); }
  Tensor sin() { return EigenTensor(tensor.unaryExpr(Eigen::internal::scalar_sin_op<EigenScalar>())); }
  Tensor cos() { return EigenTensor(tensor.unaryExpr(Eigen::internal::scalar_cos_op<EigenScalar>())); }
  Tensor tan() { return EigenTensor(tensor.unaryExpr(Eigen::internal::scalar_tan_op<EigenScalar>())); }
  Tensor atan() { return EigenTensor(tensor.unaryExpr(Eigen::internal::scalar_atan_op<EigenScalar>())); }
  Tensor asin() { return EigenTensor(tensor.unaryExpr(Eigen::internal::scalar_asin_op<EigenScalar>())); }
  Tensor acos() { return EigenTensor(tensor.unaryExpr(Eigen::internal::scalar_acos_op<EigenScalar>())); }
  Tensor tanh() { return EigenTensor(tensor.unaryExpr(Eigen::internal::scalar_tanh_op<EigenScalar>())); }
  Tensor cosh() { return EigenTensor(tensor.unaryExpr(Eigen::internal::scalar_cosh_op<EigenScalar>())); }
  Tensor sinh() { return EigenTensor(tensor.unaryExpr(Eigen::internal::scalar_sinh_op<EigenScalar>())); }
  Tensor acosh() { throw NotImplemented("No Eigen acosh support."); return EigenTensor({0,0,0}); }
  Tensor exp() { return EigenTensor(tensor.exp()); }
  Tensor log() { return EigenTensor(tensor.log()); }

  // operators - unary element wise + constant
  Tensor pow(double exponent)  { return EigenTensor(tensor.pow(exponent)); }

  // operators - reduction
  Tensor sum() { return EigenTensor(tensor.sum()); }
  Tensor sum(const Dimensions& dims) { return EigenTensor(tensor.sum(dims)); }
  Tensor max() { return EigenTensor(tensor.maximum()); }
  Tensor max(const Dimensions& dims) { return EigenTensor(tensor.maximum(dims)); }
  Tensor min() { return EigenTensor(tensor.minimum()); }
  Tensor min(const Dimensions& dims) { return EigenTensor(tensor.minimum(dims)); }
  Tensor prod() { return EigenTensor(tensor.prod()); }
  Tensor prod(const Dimensions& dims) { return EigenTensor(tensor.prod(dims)); }

  // operators - geometrical
  Tensor reshape(const Dimensions& dims) { return EigenTensor(tensor.reshape(dims)); }
  Tensor transpose(const Dimensions& dims) { return EigenTensor(tensor.shuffle(dims)); }
  Tensor slice(const StartIndices& offsets, const Sizes& extends) { 
    return EigenTensor(tensor.slice(offsets,extends));
  }
  Tensor reverse(const ReverseDimensions& dims) { 
    return EigenTensor(tensor.reverse(dims));
  }
  Tensor repeat(const Dimensions& dims) {
    return EigenTensor(tensor.broadcast(dims));
  }

  // operators - binary element wise
  Tensor operator+(const EigenTensor& other) {
    return EigenTensor(tensor+other.tensor);
  }
  Tensor operator-(const EigenTensor& other) {
    return EigenTensor(tensor-other.tensor);
  }
  Tensor operator*(const EigenTensor& other) {
    return EigenTensor(tensor*other.tensor);
  }
  Tensor operator/(const EigenTensor& other) {
    return EigenTensor(tensor/other.tensor);
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
