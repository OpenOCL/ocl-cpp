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
#ifndef OCLCPP_OCL_EIGENTENSOR_H_
#define OCLCPP_OCL_EIGENTENSOR_H_

#include <cmath>
#include <functional>
#include <algorithm>
#include <vector>
#include <string>
#include <iostream>

#include "tree_tensor/matrix.h"
#include "typedefs.h"
#include "exceptions.h"



namespace ocl
{

// Tensor class is dependent of Matrix type M.
class Tensor
{
public:

  // Use shortcut T for Tensor and M for Matrix
  typedef Matrix M;
  typedef Tensor T;

  // static constructors
  static Tensor Zeros(int rows, int cols);
  static Tensor Ones(int rows, int cols);

  // Constructors
  Tensor() { }
  Tensor(double v) { this->insert(Matrix(v)); }
  Tensor(const Matrix& m) { this->insert(m); }

  // Returns the underlying value
  std::vector<M> value();

  void disp()
  {
    std::cout << "{" << std::endl;
    for (unsigned int i=0; i < data.size(); i++) {
      std::cout << this->get(i).data() << std::endl << std::endl;
    }
    std::cout << "}" << std::endl;
  }

  Matrix get(int i) const {
    return this->data[i];
  }

  void insert(const Matrix& m) {
    this->data.push_back(m);
  }

  int length() const {
    return this->data.size();
  }

  // Slices value
  std::vector<M> slice();

  //
  // Declare tensor operations

  // operators - unary element wise
  T uplus() const;
  T uminus() const;
  T square() const;
  T inverse() const;
  T abs() const;
  T sqrt() const;
  T sin() const;
  T cos() const;
  T tan() const;
  T atan() const;
  T asin() const;
  T acos() const;
  T tanh() const;
  T cosh() const;
  T sinh() const;
  T exp() const;
  T log() const;

  // operators - unary element wise + scalar
  T pow(Scalar exponent) const;

  // reduction operations
  T norm() const;
  T sum() const;
  T min() const;
  T max() const;
  T trace() const;
  T mean() const;

  // geometrical operations
  T transpose() const;

  T reshape(Integer cols, Integer rows) const;

  // get slice (i:j)
  T slice(Integer i, Integer j) const;

  // get block slice of cols (i:j) and rows (k:l)
  T block(Integer i, Integer j, Integer k, Integer l) const;

  // binary coefficient wise
  T plus(const T& other) const;
  T minus(const T& other) const;
  T ctimes(const T& other) const;
  T cdivide(const T& other) const;

  // binary matrix operations
  T times(const T& other) const;
  T cross(const T& other) const;
  T dot(const T& other) const;

  // operator overloading
  T operator+(const T& other) const;
  T operator-(const T& other) const;
  T operator*(const T& other) const;
  T operator/(const T& other) const;

private:
  std::vector<M> data;

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
typedef Matrix (*UnaryOpFcnWithScalar)(const Matrix& m, Scalar s);
typedef Matrix (*UnaryOpFcnWithInteger2)(const Matrix& m, Integer s1, Integer s2);
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

static inline Tensor unaryVecOperationWithScalar(const Tensor& tensor, UnaryOpFcnWithScalar fcn_ptr, Scalar s)
{
  Tensor t = Tensor();
  for(unsigned int i=0; i<tensor.length(); i++) {
    t.insert( fcn_ptr(tensor.get(i), s) );
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

static inline Tensor pow(const Tensor& t, const Scalar exponent) {
  Matrix (*f)(const Matrix&, const Scalar) = &ocl::pow;
  return tensor::unaryVecOperationWithScalar(t, f, exponent);
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
static inline Tensor slice(const Tensor& t, Integer i, Integer j) {
  return tensor::unaryVecOperationWithInteger2(t, &ocl::slice, i, j);
}

// get block slice of cols (i:j) and rows (k:l)
static inline Tensor block(const Tensor& t, Integer i, Integer j, Integer k, Integer l) {
  return tensor::unaryVecOperationWithInteger4(t, &ocl::block, i, j, k, l);
}

// binary coefficient wise
static inline Tensor plus(const Tensor& t1, const Tensor& t2) { return tensor::binaryVecOperation(t1, &ocl::plus, t2); }
static inline Tensor minus(const Tensor& t1, const Tensor& t2) { return tensor::binaryVecOperation(t1, &ocl::minus, t2); }
static inline Tensor ctimes(const Tensor& t1, const Tensor& t2) { return tensor::binaryVecOperation(t1, &ocl::ctimes, t2); }
static inline Tensor cdivide(const Tensor& t1, const Tensor& t2) { return tensor::binaryVecOperation(t1, &ocl::cdiv, t2); }

// binary matrix operations
static inline Tensor times(const Tensor& t1, const Tensor& t2) { return tensor::binaryVecOperation(t1, &ocl::times, t2); }
static inline Tensor cross(const Tensor& t1, const Tensor& t2) { return tensor::binaryVecOperation(t1, &ocl::cross, t2); }
static inline Tensor dot(const Tensor& t1, const Tensor& t2) { return tensor::binaryVecOperation(t1, &ocl::dot, t2); }


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
inline Tensor Tensor::pow(const Scalar exponent) const { return ocl::pow(*this, exponent); }

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
inline Tensor Tensor::slice(Integer i, Integer j) const {
  return ocl::slice(*this, i, j);
}

// get block slice of cols (i:j) and rows (k:l)
inline Tensor Tensor::block(Integer i, Integer j, Integer k, Integer l) const {
  return ocl::block(*this, i, j, k, l);
}

// binary coefficient wise
inline Tensor Tensor::plus(const Tensor& other) const { return ocl::plus(*this, other); }
inline Tensor Tensor::minus(const Tensor& other) const { return ocl::minus(*this, other); }
inline Tensor Tensor::ctimes(const Tensor& other) const { return ocl::ctimes(*this, other); }
inline Tensor Tensor::cdivide(const Tensor& other) const { return ocl::cdivide(*this, other); }

// binary matrix operations
inline Tensor Tensor::times(const Tensor& other) const { return ocl::times(*this, other); }
inline Tensor Tensor::cross(const Tensor& other) const { return ocl::cross(*this, other); }
inline Tensor Tensor::dot(const Tensor& other) const { return ocl::dot(*this, other); }

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
#endif  // OCLCPP_OCL_EIGENTENSOR_H_
