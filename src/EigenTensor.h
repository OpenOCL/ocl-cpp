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
const int R = 3;
typedef Eigen::Tensor<float, R> EigenTensorRX;
typedef EigenTensorRX::Dimensions Dimensions;
typedef float EigenScalar;
typedef Eigen::array<Eigen::DenseIndex, R> StartIndices;
typedef Eigen::array<Eigen::DenseIndex, R> Sizes;
typedef Eigen::array<bool, R> ReverseDimensions;

// Base class for tensors.
// Operators and constant tensors derive from this class.
class TensorOperator
{
public:
  virtual EigenTensorRX eval() const = 0;
};

// Constant tensor.
class Tensor : TensorOperator
{
public:
  TensorConstant(EigenTensorRX v) : v(v) { }
  virtual EigenTensorRX eval() const override
  {
    return v;
  }
private:
  EigenTensorRX v;
};

class UnaryOperator : public TensorOperator
{
public:
  UnaryOperator(const TensorOperator& input)
      : input(input) { }
protected:
  const TensorOperator& input;
};

class BinaryOperator : public TensorOperator
{
public:
  BinaryOperator(const TensorOperator& input1, const TensorOperator& input2)
      : input1(input1), input2(input2) { }

protected:
  const TensorOperator& input1;
  const TensorOperator& input2;
};

class Uplus : public UnaryOperator
{
public:
  Uplus(const TensorOperator& input): UnaryOperator(input) { }
  virtual EigenTensorRX eval() const override
  {
    return input.eval();
  }
};

class Plus : public BinaryOperator
{
public:
  Plus(const TensorOperator& input1, const TensorOperator& input2)
      : BinaryOperator(input1, input2) { }
  virtual EigenTensorRX eval() const override
  {
    return input1.eval() + input2.eval();
  }
};

} // namespace ocl


#endif  // OCLCPP_OCL_EIGENTENSOR_H_
