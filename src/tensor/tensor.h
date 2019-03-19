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
#ifndef OCLCPP_OCL_TENSOR_H_
#define OCLCPP_OCL_TENSOR_H_

#include <iostream>            // disp

#include "utils/typedefs.h"    // Integer
#include "tensor/matrix.h"     // Matrix
#include "utils/slicing.h"     // Slicable

namespace ocl
{

// Tensor class
class Tensor : public Slicable
{
public:

  // static constructors
  static Tensor Zero(const int rows, const int cols) {
    return Tensor(Matrix::Zero(rows, cols));
  }

  static Tensor One(const int rows, const int cols) {
    return Tensor(Matrix::One(rows, cols));
  };

  // Constructors
  Tensor() { }
  Tensor(double v) { this->insert(Matrix(v)); }
  Tensor(const Matrix& m) { this->insert(m); }

  Tensor(const std::vector<Matrix>& m) : data(m) { }

  // size of either first or second dimension
  virtual int size(const int dim) const {
    return this->data[0].size(dim);
  }

  // length of Tensor (3rd dimension)
  int size() const {
    return data.size();
  }

  void disp()
  {
    std::cout << "{" << std::endl;
    for (unsigned int i=0; i < data.size(); i++) {
      std::cout << this->get(i).data() << std::endl << std::endl;
    }
    std::cout << "}" << std::endl;
  }

  Matrix get(const int i) const {
    return this->data[i];
  }

  void insert(const Matrix& m) {
    this->data.push_back(m);
  }

  unsigned int length() const {
    return this->data.size();
  }

  //
  // Declare tensor operations

  // operators - unary element wise
  Tensor uplus() const;
  Tensor uminus() const;
  Tensor square() const;
  Tensor inverse() const;
  Tensor abs() const;
  Tensor sqrt() const;
  Tensor sin() const;
  Tensor cos() const;
  Tensor tan() const;
  Tensor atan() const;
  Tensor asin() const;
  Tensor acos() const;
  Tensor tanh() const;
  Tensor cosh() const;
  Tensor sinh() const;
  Tensor exp() const;
  Tensor log() const;

  // operators - unary element wise + scalar
  Tensor cpow(const Tensor& exponent) const;

  // reduction operations
  Tensor norm() const;
  Tensor sum() const;
  Tensor min() const;
  Tensor max() const;
  Tensor trace() const;
  Tensor mean() const;

  // geometrical operations
  Tensor transpose() const;

  Tensor reshape(Integer cols, Integer rows) const;

  // get slice (i:j)
  Tensor slice(const std::vector<int>& slice1, const std::vector<int>& slice2) const;

  // binary coefficient wise
  Tensor plus(const Tensor& other) const;
  Tensor minus(const Tensor& other) const;
  Tensor ctimes(const Tensor& other) const;
  Tensor cdivide(const Tensor& other) const;

  Tensor cmin(const Tensor& other) const;
  Tensor cmax(const Tensor& other) const;

  // binary matrix operations
  Tensor times(const Tensor& other) const;
  Tensor cross(const Tensor& other) const;
  Tensor dot(const Tensor& other) const;

  Tensor atan2(const Tensor& other) const;

  // operator overloading
  Tensor operator+(const Tensor& other) const;
  Tensor operator-(const Tensor& other) const;
  Tensor operator*(const Tensor& other) const;
  Tensor operator/(const Tensor& other) const;

private:
  std::vector<Matrix> data;

}; // class Tensor<M>


std::vector<double> full(const Tensor& t) {
  return ocl::full(t.get(0));
}

namespace tensor
{
// static functions

//
// General functions to operate on vector of matrizes

// Function pointers to static functions
typedef Matrix (*UnaryOpFcn)(const Matrix& m);
typedef Matrix (*UnaryOpFcnWithInteger2)(const Matrix& m, Integer s1, Integer s2);
typedef Matrix (*UnaryOpFcnWithIntegerVec2)(const Matrix& m, const std::vector<int>& vec1, const std::vector<int>& vec2);
typedef Matrix (*UnaryOpFcnWithInteger4)(const Matrix& m, Integer s1, Integer s2, Integer s3, Integer s4);
typedef Matrix (*UnaryReductionOpFcn)(const Matrix& m);

typedef Matrix (*BinaryOpFcn)(const Matrix& m1, const Matrix& m2);

// Apply unary operator function to all matrizes in the vector
static inline Tensor unaryVecOperation(const Tensor& tensor, UnaryOpFcn fcn_ptr)
{
  Tensor t = Tensor();
  for(unsigned int i=0; i<tensor.length(); i++) {
    t.insert( fcn_ptr(tensor.get(i)) );
  }
  return t;
}

static inline Tensor unaryVecOperationWithInteger2(const Tensor& tensor, UnaryOpFcnWithInteger2 fcn_ptr, Integer s1, Integer s2)
{
  Tensor t = Tensor();
  for(unsigned int i=0; i<tensor.length(); i++) {
    t.insert( fcn_ptr(tensor.get(i), s1, s2) );
  }
  return t;
}

static inline Tensor unaryVecOperationWithIntegerVec2(const Tensor& tensor, UnaryOpFcnWithIntegerVec2 fcn_ptr, const std::vector<int>& vec1, const std::vector<int>& vec2)
{
  Tensor t = Tensor();
  for(unsigned int i=0; i<tensor.length(); i++) {
    t.insert( fcn_ptr(tensor.get(i), vec1, vec2) );
  }
  return t;
}

static inline Tensor unaryVecOperationWithInteger4(const Tensor& tensor, UnaryOpFcnWithInteger4 fcn_ptr, Integer s1, Integer s2, Integer s3, Integer s4)
{
  Tensor t = Tensor();
  for(unsigned int i=0; i<tensor.length(); i++) {
    t.insert( fcn_ptr(tensor.get(i), s1, s2, s3, s4) );
  }
  return t;
}

static inline Tensor unaryReductionOperation(const Tensor& tensor, UnaryReductionOpFcn fcn_ptr)
{
  Tensor t = Tensor();
  for(unsigned int i=0; i<tensor.length(); i++) {
    t.insert( fcn_ptr(tensor.get(i)) );
  }
  return t;
}

static inline Tensor binaryVecOperation(const Tensor& tensor, BinaryOpFcn fcn_ptr, const Tensor& other)
{
  // TODO: implement broadcasting
  //assertEqual(other.data.size(), 1);

  Tensor t = Tensor();
  for(unsigned int i=0; i<tensor.length(); i++) {
    t.insert( fcn_ptr(tensor.get(i), other.get(0)) );
  }
  return t;
}
} // namespace tensor

// static operator functions (calling vec operation with function pointer to Matrix functions)
static inline Tensor uplus(const Tensor& t) {
  Matrix (*f)(const Matrix&) = &ocl::uplus;
  return tensor::unaryVecOperation(t, f);
}
static inline Tensor uminus(const Tensor& t) {
  Matrix (*f)(const Matrix&) = &ocl::uminus;
  return tensor::unaryVecOperation(t, f);
}
static inline Tensor square(const Tensor& t) {
  Matrix (*f)(const Matrix&) = &ocl::square;
  return tensor::unaryVecOperation(t, f);
}
static inline Tensor inverse(const Tensor& t) {
  Matrix (*f)(const Matrix&) = &ocl::inverse;
  return tensor::unaryVecOperation(t, f);
}
static inline Tensor abs(const Tensor& t) {
  Matrix (*f)(const Matrix&) = &ocl::abs;
  return tensor::unaryVecOperation(t, f);
}
static inline Tensor sqrt(const Tensor& t) {
  Matrix (*f)(const Matrix&) = &ocl::sqrt;
  return tensor::unaryVecOperation(t, f);
}
static inline Tensor sin(const Tensor& t) {
  Matrix (*f)(const Matrix&) = &ocl::sin;
  return tensor::unaryVecOperation(t, f);
}
static inline Tensor cos(const Tensor& t) {
  Matrix (*f)(const Matrix&) = &ocl::cos;
  return tensor::unaryVecOperation(t, f);
}
static inline Tensor tan(const Tensor& t) {
  Matrix (*f)(const Matrix&) = &ocl::tan;
  return tensor::unaryVecOperation(t, f);
}
static inline Tensor atan(const Tensor& t) {
  Matrix (*f)(const Matrix&) = &ocl::atan;
  return tensor::unaryVecOperation(t, f);
}
static inline Tensor asin(const Tensor& t) {
  Matrix (*f)(const Matrix&) = &ocl::asin;
  return tensor::unaryVecOperation(t, f);
}
static inline Tensor acos(const Tensor& t) {
  Matrix (*f)(const Matrix&) = &ocl::acos;
  return tensor::unaryVecOperation(t, f);
}
static inline Tensor tanh(const Tensor& t) {
  Matrix (*f)(const Matrix&) = &ocl::tanh;
  return tensor::unaryVecOperation(t, f);
}
static inline Tensor cosh(const Tensor& t) {
   Matrix (*f)(const Matrix&) = &ocl::cosh;
   return tensor::unaryVecOperation(t, f);
 }
static inline Tensor sinh(const Tensor& t) {
  Matrix (*f)(const Matrix&) = &ocl::sinh;
  return tensor::unaryVecOperation(t, f);
}
static inline Tensor exp(const Tensor& t) {
  Matrix (*f)(const Matrix&) = &ocl::exp;
  return tensor::unaryVecOperation(t, f);
}
static inline Tensor log(const Tensor& t) {
  Matrix (*f)(const Matrix&) = &ocl::log;
  return tensor::unaryVecOperation(t, f);
}

static inline Tensor cpow(const Tensor& t, const Tensor& exponent) {
  Matrix (*f)(const Matrix&, const Matrix&) = &ocl::cpow;
  return tensor::binaryVecOperation(t, f, exponent);
}

static inline Tensor norm(const Tensor& t) {
  Matrix (*f)(const Matrix&) = &ocl::norm;
  return tensor::unaryVecOperation(t, f);
}

static inline Tensor sum(const Tensor& t) {
  Matrix (*f)(const Matrix&) = &ocl::sum;
  return tensor::unaryVecOperation(t, f);
}

static inline Tensor min(const Tensor& t) {
  Matrix (*f)(const Matrix&) = &ocl::min;
  return tensor::unaryVecOperation(t, f);
}

static inline Tensor max(const Tensor& t) {
  Matrix (*f)(const Matrix&) = &ocl::max;
  return tensor::unaryVecOperation(t, f);
}

static inline Tensor trace(const Tensor& t) {
  Matrix (*f)(const Matrix&) = &ocl::trace;
  return tensor::unaryVecOperation(t, f);
}

static inline Tensor mean(const Tensor& t) {
  Matrix (*f)(const Matrix&) = &ocl::mean;
  return tensor::unaryVecOperation(t, f);
}

static inline Tensor transpose(const Tensor& t) {
  return tensor::unaryVecOperation(t, &ocl::transpose);
}

static inline Tensor reshape(const Tensor& t, Integer cols, Integer rows) {
  return tensor::unaryVecOperationWithInteger2(t, &ocl::reshape, cols, rows);
}

// get slice (i:j)
static inline Tensor slice(const Tensor& t, const std::vector<int>& slice1, const std::vector<int>& slice2) {
  return tensor::unaryVecOperationWithIntegerVec2(t, &ocl::slice, slice1, slice2);
}


// binary coefficient wise
static inline Tensor plus(const Tensor& t1, const Tensor& t2) { return tensor::binaryVecOperation(t1, &ocl::plus, t2); }
static inline Tensor minus(const Tensor& t1, const Tensor& t2) { return tensor::binaryVecOperation(t1, &ocl::minus, t2); }
static inline Tensor ctimes(const Tensor& t1, const Tensor& t2) { return tensor::binaryVecOperation(t1, &ocl::ctimes, t2); }
static inline Tensor cdivide(const Tensor& t1, const Tensor& t2) { return tensor::binaryVecOperation(t1, &ocl::cdivide, t2); }

static inline Tensor cmin(const Tensor& t1, const Tensor& t2) { return tensor::binaryVecOperation(t1, &ocl::cmin, t2); }
static inline Tensor cmax(const Tensor& t1, const Tensor& t2) { return tensor::binaryVecOperation(t1, &ocl::cmax, t2); }

// binary matrix operations
static inline Tensor times(const Tensor& t1, const Tensor& t2) { return tensor::binaryVecOperation(t1, &ocl::times, t2); }
static inline Tensor cross(const Tensor& t1, const Tensor& t2) { return tensor::binaryVecOperation(t1, &ocl::cross, t2); }
static inline Tensor dot(const Tensor& t1, const Tensor& t2) { return tensor::binaryVecOperation(t1, &ocl::dot, t2); }

static inline Tensor atan2(const Tensor& t1, const Tensor& t2) { return tensor::binaryVecOperation(t1, &ocl::atan2, t2); }


//
// Define Tensor operations

// operators - unary element wise
inline Tensor Tensor::uplus() const { return ocl::uplus(*this); }
inline Tensor Tensor::uminus() const { return ocl::uminus(*this); }
inline Tensor Tensor::square() const { return ocl::square(*this); }
inline Tensor Tensor::inverse() const { return ocl::inverse(*this); }
inline Tensor Tensor::abs() const { return ocl::abs(*this); }
inline Tensor Tensor::sqrt() const { return ocl::sqrt(*this); }
inline Tensor Tensor::sin() const { return ocl::sin(*this); }
inline Tensor Tensor::cos() const { return ocl::cos(*this); }
inline Tensor Tensor::tan() const { return ocl::tan(*this); }
inline Tensor Tensor::atan() const { return ocl::atan(*this); }
inline Tensor Tensor::asin() const { return ocl::asin(*this); }
inline Tensor Tensor::acos() const { return ocl::acos(*this); }
inline Tensor Tensor::tanh() const { return ocl::tanh(*this); }
inline Tensor Tensor::cosh() const { return ocl::cosh(*this); }
inline Tensor Tensor::sinh() const { return ocl::sinh(*this); }
inline Tensor Tensor::exp() const { return ocl::exp(*this); }
inline Tensor Tensor::log() const { return ocl::log(*this); }

// operators - unary element wise + scalar
inline Tensor Tensor::cpow(const Tensor& exponent) const { return ocl::cpow(*this, exponent); }

// reduction operations
inline Tensor Tensor::norm() const { return ocl::norm(*this); }
inline Tensor Tensor::sum() const { return ocl::sum(*this); }
inline Tensor Tensor::min() const { return ocl::min(*this); }
inline Tensor Tensor::max() const { return ocl::max(*this); }
inline Tensor Tensor::trace() const { return ocl::trace(*this); }
inline Tensor Tensor::mean() const { return ocl::mean(*this); }

// geometrical operations
inline Tensor Tensor::transpose() const { return ocl::transpose(*this); }

inline Tensor Tensor::reshape(Integer cols, Integer rows) const {
  return ocl::reshape(*this, cols, rows);
}

// get slice (i:j)
inline Tensor Tensor::slice(const std::vector<int>& slice1, const std::vector<int>& slice2) const {
  return ocl::slice(*this, slice1, slice2);
}

// binary coefficient wise
inline Tensor Tensor::plus(const Tensor& other) const { return ocl::plus(*this, other); }
inline Tensor Tensor::minus(const Tensor& other) const { return ocl::minus(*this, other); }
inline Tensor Tensor::ctimes(const Tensor& other) const { return ocl::ctimes(*this, other); }
inline Tensor Tensor::cdivide(const Tensor& other) const { return ocl::cdivide(*this, other); }

inline Tensor Tensor::cmin(const Tensor& other) const { return ocl::cmin(*this, other); }
inline Tensor Tensor::cmax(const Tensor& other) const { return ocl::cmax(*this, other); }

// binary matrix operations
inline Tensor Tensor::times(const Tensor& other) const { return ocl::times(*this, other); }
inline Tensor Tensor::cross(const Tensor& other) const { return ocl::cross(*this, other); }
inline Tensor Tensor::dot(const Tensor& other) const { return ocl::dot(*this, other); }

inline Tensor Tensor::atan2(const Tensor& other) const { return ocl::atan2(*this, other); }

// operator overloading
inline Tensor Tensor::operator+(const Tensor& other) const {
  return this->plus(other);
}
inline Tensor Tensor::operator-(const Tensor& other) const {
  return this->minus(other);
}
inline Tensor Tensor::operator*(const Tensor& other) const {
  return this->times(other);
}
inline Tensor Tensor::operator/(const Tensor& other) const {
  return this->cdivide(other);
}

} // namespace ocl
#endif  // OCLCPP_OCL_TENSOR_H_
