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

#include <Eigen/Geometry>

#include "typedefs.h"
#include "exceptions.h"
#include "NumericMatrix.h"


namespace ocl
{

// forward declarations (defined below Tensor class)
template<class M>
class Tensor;

template<class M>
static inline Tensor<M> cos(const Tensor<M>& t);

// Tensor class is dependent of Matrix type M.
template<class M>
class Tensor
{
  // Use shortcut T for Tensor<M>
  typedef Tensor<M> T;
  typedef float Scalar;
  typedef int Integer;

 public:



  // static constructors
  static T FromMatrix(const M& mat) {
    T t = T();
    t.data = {mat};
    return t;
  }

  // Constructor
  Tensor<M>(Eigen::Index rows, Eigen::Index cols) : data(1, M::Zero(rows,cols)) { }
  Tensor<M>() { }

  // Returns the underlying value
  std::vector<M> value();

  // Return a string representation
  std::string str();

  // Sets a value, supports broadcasting
  // void set(std::initializer_list<std::initializer_list<double> > values) { tensor.setValues(values); }

  // Slices value
  std::vector<M> slice();

  //
  // General functions to operate on vector of matrizes

  // Function pointers to static functions
  typedef M (*UnaryOpFcn)(const M& m);
  typedef M (*UnaryOpFcnWithScalar)(const M& m, Scalar s);
  typedef M (*UnaryOpFcnWithInteger2)(const M& m, Integer s1, Integer s2);
  typedef M (*UnaryOpFcnWithInteger4)(const M& m, Integer s1, Integer s2, Integer s3, Integer s4);
  typedef M (*UnaryReductionOpFcn)(const M& m);

  typedef M (*BinaryOpFcn)(const M& m1, const M& m2);


  // Apply unary operator function to all matrizes in the vector
  static T unaryVecOperation(const T& tensor, UnaryOpFcn fcn_ptr)
  {
    T t = T();
    for(unsigned int i=0; i<tensor.data.size(); i++) {
      t.data.push_back( fcn_ptr(tensor.data[i]) );
    }
    return t;
  }

  static T unaryVecOperationWithScalar(const T& tensor, UnaryOpFcnWithScalar fcn_ptr, Scalar s)
  {
    T t = T();
    for(unsigned int i=0; i<tensor.data.size(); i++) {
      t.data.push_back( fcn_ptr(tensor.data[i], s) );
    }
    return t;
  }

  static T unaryVecOperationWithInteger2(const T& tensor, UnaryOpFcnWithInteger2 fcn_ptr, Integer s1, Integer s2)
  {
    T t = T();
    for(unsigned int i=0; i<tensor.data.size(); i++) {
      t.data.push_back( fcn_ptr(tensor.data[i], s1, s2) );
    }
    return t;
  }

  static T unaryVecOperationWithInteger4(const T& tensor, UnaryOpFcnWithInteger4 fcn_ptr, Integer s1, Integer s2, Integer s3, Integer s4)
  {
    T t = T();
    for(unsigned int i=0; i<tensor.data.size(); i++) {
      t.data.push_back( fcn_ptr(tensor.data[i], s1, s2, s3, s4) );
    }
    return t;
  }

  static T unaryReductionOperation(const T& tensor, UnaryReductionOpFcn fcn_ptr)
  {
    T t = Tensor();
    for(unsigned int i=0; i<tensor.data.size(); i++) {
      Scalar s = fcn_ptr(tensor.data[i]);
      M m;
      m << s;
      t.data.push_back( m );
    }
    return t;
  }

  static T binaryVecOperation(const T& tensor, BinaryOpFcn fcn_ptr, const T& other)
  {
    // TODO: implement broadcasting
    //assertEqual(other.data.size(), 1);

    T t = T();
    for(unsigned int i=0; i<tensor.data.size(); i++) {
      t.data.push_back( fcn_ptr(tensor.data[i], other.data[0]) );
    }
    return t;
  }


  //
  // Define tensor operations

  // operators - unary element wise
  T uplus() const { return unaryVecOperation(&M::uplus); }
  T uminus() const { return unaryVecOperation(&M::uminus); }
  T square() const { return unaryVecOperation(&M::square); }
  T inverse() const { return unaryVecOperation(&M::inverse); }
  T abs() const { return unaryVecOperation(&M::abs); }
  T sqrt() const { return unaryVecOperation(&M::sqrt); }
  T sin() const { return unaryVecOperation(&M::sin); }
  T cos() const { return cos(*this); }
  T tan() const { return unaryVecOperation(&M::tan); }
  T atan() const { return unaryVecOperation(&M::atan); }
  T asin() const { return unaryVecOperation(&M::asin); }
  T acos() const { return unaryVecOperation(&M::acos); }
  T tanh() const { return unaryVecOperation(&M::tanh); }
  T cosh() const { return unaryVecOperation(&M::cosh); }
  T sinh() const { return unaryVecOperation(&M::sinh); }
  T exp() const { return unaryVecOperation(&M::exp); }
  T log() const { return unaryVecOperation(&M::log); }

  // operators - unary element wise + scalar
  T pow(M exponent) const {
    return unaryVecOperationWithScalar(&M::pow, exponent);
  }

  // reduction operations
  T norm() const { return unaryReductionOperation(&M::norm); }
  T sum() const { return unaryReductionOperation(&M::sum); }
  T min() const { return unaryReductionOperation(&M::min); }
  T max() const { return unaryReductionOperation(&M::max); }
  T trace() const { return unaryReductionOperation(&M::trace); }
  T mean() const { return unaryReductionOperation(&M::mean); }
  T prod() const { return unaryReductionOperation(&M::prod); }

  // geometrical operations
  T transpose() const { return unaryVecOperation(&M::transpose); }

  T reshape(Integer cols, Integer rows) const {
    return unaryVecOperationWithInteger2(&M::reshape, cols, rows);
  }

  // get slice (i:j)
  T slice(Integer i, Integer j) const {
    return unaryVecOperationWithInteger2(&M::slice, i, j);
  }

  // get block slice of cols (i:j) and rows (k:l)
  T slice(Integer i, Integer j, Integer k, Integer l) const {
    return unaryVecOperationWithInteger4(&M::block, i, j, k, l);
  }

  // binary coefficient wise
  T plus(const T& other) const { return binaryVecOperation(&M::cplus, other); }
  T minus(const T& other) const { return binaryVecOperation(&M::cminus, other); }
  T ctimes(const T& other) const { return binaryVecOperation(&M::ctimes, other); }
  T cdivide(const T& other) const { return binaryVecOperation(&M::cdiv, other); }

  // binary matrix operations
  T times(const T& other) const { return binaryVecOperation(&M::times, other); }
  T cross(const T& other) const { return binaryVecOperation(&M::cross, other); }
  T dot(const T& other) const { return binaryVecOperation(&M::dot, other); }

  // operator overloading
  T operator+(const T& other) {
    return this->plus(other);
  }
  T operator-(const T& other) {
    return this->minus(other);
  }
  T operator*(const T& other) {
    return this->times(other);
  }
  T operator/(const T& other) {
    return this->cdivide(other);
  }

 private:
  std::vector<M> data;

}; // class Tensor<M>

template<class M>
static inline Tensor<M> cos(const Tensor<M>& t) { return Tensor<M>::unaryVecOperation(t, &M::cos); }


} // namespace ocl
#endif  // OCLCPP_OCL_EIGENTENSOR_H_
