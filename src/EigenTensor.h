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


namespace ocl
{

// Tensor class for 3rd order tensors represented as a vector matrizes
class Tensor
{

 public:
  typedef Eigen::MatrixXf EigenMatrix;
  typedef Eigen::Vector3f EigenVector3;
  typedef Eigen::VectorXf EigenVector;
  typedef float Scalar;
  typedef int Integer;

  // static constructors
  Tensor FromEigenMatrix(const EigenMatrix& mat) {
    Tensor t = Tensor();
    t.tensor = {mat};
    return t;
  }

  // Constructor
  Tensor(Eigen::Index rows, Eigen::Index cols) : tensor(1, EigenMatrix::Zero(rows,cols)) { }
  Tensor() { }

  // Returns the underlying value
  std::vector<EigenMatrix> value();

  // Return a string representation
  std::string str();

  // Sets a value, supports broadcasting
  // void set(std::initializer_list<std::initializer_list<double> > values) { tensor.setValues(values); }

  // Slices value
  std::vector<EigenMatrix> slice();

  // linspace operator
  // EigenTensorRX linspace(const Tensor& other);


  // General functions to operate on vector of matrizes

  // Function pointers to static functions
  typedef EigenMatrix (*UnaryOpFcn)(const EigenMatrix& m);
  typedef EigenMatrix (*UnaryOpFcnWithScalar)(const EigenMatrix& m, Scalar s);
  typedef EigenMatrix (*UnaryOpFcnWithInteger2)(const EigenMatrix& m, Integer s1, Integer s2);
  typedef EigenMatrix (*UnaryOpFcnWithInteger4)(const EigenMatrix& m, Integer s1, Integer s2, Integer s3, Integer s4);
  typedef Scalar (*UnaryReductionOpFcn)(const EigenMatrix& m);

  typedef EigenMatrix (*BinaryOpFcn)(const EigenMatrix& m1, const EigenMatrix& m2);

  // Apply unary operator function to all matrizes in the vector
  Tensor unaryVecOperation(UnaryOpFcn fcn_ptr) const
  {
    Tensor t = Tensor();
    for(unsigned int i=0; i<tensor.size(); i++) {
      t.tensor.push_back( fcn_ptr(tensor[i]) );
    }
    return t;
  }

  Tensor unaryVecOperationWithScalar(UnaryOpFcnWithScalar fcn_ptr, Scalar s) const
  {
    Tensor t = Tensor();
    for(unsigned int i=0; i<tensor.size(); i++) {
      t.tensor.push_back( fcn_ptr(tensor[i], s) );
    }
    return t;
  }

  Tensor unaryVecOperationWithInteger2(UnaryOpFcnWithInteger2 fcn_ptr, Integer s1, Integer s2) const
  {
    Tensor t = Tensor();
    for(unsigned int i=0; i<tensor.size(); i++) {
      t.tensor.push_back( fcn_ptr(tensor[i], s1, s2) );
    }
    return t;
  }

  Tensor unaryVecOperationWithInteger4(UnaryOpFcnWithInteger4 fcn_ptr, Integer s1, Integer s2, Integer s3, Integer s4) const
  {
    Tensor t = Tensor();
    for(unsigned int i=0; i<tensor.size(); i++) {
      t.tensor.push_back( fcn_ptr(tensor[i], s1, s2, s3, s4) );
    }
    return t;
  }

  Tensor unaryReductionOperation(UnaryReductionOpFcn fcn_ptr) const
  {
    Tensor t = Tensor();
    for(unsigned int i=0; i<tensor.size(); i++) {
      Scalar s = fcn_ptr(tensor[i]);
      EigenMatrix m;
      m << s;
      t.tensor.push_back( m );
    }
    return t;
  }

  Tensor binaryVecOperation(BinaryOpFcn fcn_ptr, const Tensor& other) const
  {
    // TODO: implement broadcasting
    //assertEqual(other.tensor.size(), 1);

    Tensor t = Tensor();
    for(unsigned int i=0; i<tensor.size(); i++) {
      t.tensor.push_back( fcn_ptr(tensor[i], other.tensor[0]) );
    }
    return t;
  }

  //
  // Define functions on matrizes

  // unary element wise
  static EigenMatrix m_uplus(const EigenMatrix& m) { return m; }
  static EigenMatrix m_uminus(const EigenMatrix& m) { return -m; }
  static EigenMatrix m_square(const EigenMatrix& m) { return Eigen::square(m.array()); }
  static EigenMatrix m_inverse(const EigenMatrix& m) { return Eigen::inverse(m.array()); }
  static EigenMatrix m_abs(const EigenMatrix& m) { return Eigen::abs(m.array()); }
  static EigenMatrix m_sqrt(const EigenMatrix& m) { return Eigen::sqrt(m.array()); }
  static EigenMatrix m_sin(const EigenMatrix& m) { return Eigen::sin(m.array()); }
  static EigenMatrix m_cos(const EigenMatrix& m) { return Eigen::cos(m.array()); }
  static EigenMatrix m_tan(const EigenMatrix& m) { return Eigen::tan(m.array()); }
  static EigenMatrix m_atan(const EigenMatrix& m) { return Eigen::atan(m.array()); }
  static EigenMatrix m_asin(const EigenMatrix& m) { return Eigen::asin(m.array()); }
  static EigenMatrix m_acos(const EigenMatrix& m) { return Eigen::acos(m.array()); }
  static EigenMatrix m_tanh(const EigenMatrix& m) { return Eigen::tanh(m.array()); }
  static EigenMatrix m_sinh(const EigenMatrix& m) { return Eigen::sinh(m.array()); }
  static EigenMatrix m_cosh(const EigenMatrix& m) { return Eigen::cosh(m.array()); }
  static EigenMatrix m_exp(const EigenMatrix& m) { return Eigen::exp(m.array()); }
  static EigenMatrix m_log(const EigenMatrix& m) { return Eigen::log(m.array()); }

  // unary element wise + constant
  static EigenMatrix m_pow(const EigenMatrix& m, const Scalar exponent) { return m.array().pow(exponent); }

  // reduction
  static Scalar m_norm(const EigenMatrix& m) { return m.norm(); }
  static Scalar m_sum(const EigenMatrix& m) { return m.sum(); }
  static Scalar m_min(const EigenMatrix& m) { return m.minCoeff(); }
  static Scalar m_max(const EigenMatrix& m) { return m.maxCoeff(); }
  static Scalar m_mean(const EigenMatrix& m) { return m.mean(); }
  static Scalar m_trace(const EigenMatrix& m) { return m.trace(); }
  static Scalar m_prod(const EigenMatrix& m) { return m.prod(); }

  // geometrical
  static EigenMatrix m_reshape(const EigenMatrix& m, int rows, int cols) {
    // create copy of matrix data, and create map with new dimensions but same data
    Scalar data_copy[m.size()];
    std::copy(m.data(), m.data() + m.size(), data_copy);
    Eigen::Map<EigenMatrix> res(data_copy, rows, cols);
    return res;
  }
  static EigenMatrix m_transpose(const EigenMatrix& m) { return m.transpose(); }
  // get block slice of cols (i:j) and rows (k:l)
  static EigenMatrix m_block(const EigenMatrix& m, int i, int j, int k, int l) { return m.block(i,k-i+1,j,l-j+1); }
  // get element at (i,k)
  static EigenMatrix m_slice(const EigenMatrix& m, int i, int j) { return m.block(i,j,1,1); }

  // binary coefficient wise
  static EigenMatrix m_ctimes(const EigenMatrix& m1, const EigenMatrix& m2) { return m1.array() * m2.array(); }
  static EigenMatrix m_cplus(const EigenMatrix& m1, const EigenMatrix& m2) { return m1.array() + m2.array(); }
  static EigenMatrix m_cdiv(const EigenMatrix& m1, const EigenMatrix& m2) { return m1.array() / m2.array(); }
  static EigenMatrix m_cminus(const EigenMatrix& m1, const EigenMatrix& m2) { return m1.array() - m2.array(); }

  // binary operations
  static EigenMatrix m_times(const EigenMatrix& m1, const EigenMatrix& m2) { return m1 * m2; }

  static EigenMatrix m_cross(const EigenMatrix& m1, const EigenMatrix& m2) {
    // convert matrizes to vectors of length 3
    Scalar m1_data[m1.size()];
    std::copy(m1.data(), m1.data() + m1.size(), m1_data);

    Scalar m2_data[m2.size()];
    std::copy(m2.data(), m2.data() + m2.size(), m2_data);

    Eigen::Map<EigenVector3> v1(m1_data);
    Eigen::Map<EigenVector3> v2(m2_data);
    return v1.cross(v2);
  }

  static EigenMatrix m_dot(const EigenMatrix& m1, const EigenMatrix& m2) {
    // convert matrizes to vectors
    Scalar m1_data[m1.size()];
    std::copy(m1.data(), m1.data() + m1.size(), m1_data);

    Scalar m2_data[m2.size()];
    std::copy(m2.data(), m2.data() + m2.size(), m2_data);

    Eigen::Map<EigenVector> v1(m1_data, m1.size());
    Eigen::Map<EigenVector> v2(m2_data, m2.size());

    Scalar s = v1.dot(v2);
    EigenMatrix m;
    m << s;
    return m;
  }

  //
  // Define tensor operations

  // operators - unary element wise
  Tensor uplus() const { return unaryVecOperation(&Tensor::m_uplus); }
  Tensor uminus() const { return unaryVecOperation(&Tensor::m_uminus); }
  Tensor square() const { return unaryVecOperation(&Tensor::m_square); }
  Tensor inverse() const { return unaryVecOperation(&Tensor::m_inverse); }
  Tensor abs() const { return unaryVecOperation(&Tensor::m_abs); }
  Tensor sqrt() const { return unaryVecOperation(&Tensor::m_sqrt); }
  Tensor sin() const { return unaryVecOperation(&Tensor::m_sin); }
  Tensor cos() const { return unaryVecOperation(&Tensor::m_cos); }
  Tensor tan() const { return unaryVecOperation(&Tensor::m_tan); }
  Tensor atan() const { return unaryVecOperation(&Tensor::m_atan); }
  Tensor asin() const { return unaryVecOperation(&Tensor::m_asin); }
  Tensor acos() const { return unaryVecOperation(&Tensor::m_acos); }
  Tensor tanh() const { return unaryVecOperation(&Tensor::m_tanh); }
  Tensor cosh() const { return unaryVecOperation(&Tensor::m_cosh); }
  Tensor sinh() const { return unaryVecOperation(&Tensor::m_sinh); }
  Tensor exp() const { return unaryVecOperation(&Tensor::m_exp); }
  Tensor log() const { return unaryVecOperation(&Tensor::m_log); }

  // operators - unary element wise + scalar
  Tensor pow(Scalar exponent) const {
    return unaryVecOperationWithScalar(&Tensor::m_pow, exponent);
  }

  // reduction operations
  Tensor norm() const { return unaryReductionOperation(&Tensor::m_norm); }
  Tensor sum() const { return unaryReductionOperation(&Tensor::m_sum); }
  Tensor min() const { return unaryReductionOperation(&Tensor::m_min); }
  Tensor max() const { return unaryReductionOperation(&Tensor::m_max); }
  Tensor trace() const { return unaryReductionOperation(&Tensor::m_trace); }
  Tensor mean() const { return unaryReductionOperation(&Tensor::m_mean); }
  Tensor prod() const { return unaryReductionOperation(&Tensor::m_prod); }

  // geometrical operations
  Tensor transpose() const { return unaryVecOperation(&Tensor::m_transpose); }

  Tensor reshape(Integer cols, Integer rows) const {
    return unaryVecOperationWithInteger2(&Tensor::m_reshape, cols, rows);
  }

  // get slice (i:j)
  Tensor slice(Integer i, Integer j) const {
    return unaryVecOperationWithInteger2(&Tensor::m_slice, i, j);
  }

  // get block slice of cols (i:j) and rows (k:l)
  Tensor slice(Integer i, Integer j, Integer k, Integer l) const {
    return unaryVecOperationWithInteger4(&Tensor::m_block, i, j, k, l);
  }

  // binary coefficient wise
  Tensor plus(const Tensor& other) const { return binaryVecOperation(&Tensor::m_cplus, other); }
  Tensor minus(const Tensor& other) const { return binaryVecOperation(&Tensor::m_cminus, other); }
  Tensor ctimes(const Tensor& other) const { return binaryVecOperation(&Tensor::m_ctimes, other); }
  Tensor cdivide(const Tensor& other) const { return binaryVecOperation(&Tensor::m_cdiv, other); }

  // binary matrix operations
  Tensor times(const Tensor& other) const { return binaryVecOperation(&Tensor::m_times, other); }
  Tensor cross(const Tensor& other) const { return binaryVecOperation(&Tensor::m_cross, other); }
  Tensor dot(const Tensor& other) const { return binaryVecOperation(&Tensor::m_dot, other); }

  // operator overloading
  Tensor operator+(const Tensor& other) {
    return this->plus(other);
  }
  Tensor operator-(const Tensor& other) {
    return this->minus(other);
  }
  Tensor operator*(const Tensor& other) {
    return this->times(other);
  }
  Tensor operator/(const Tensor& other) {
    return this->cdivide(other);
  }

 private:
  std::vector<EigenMatrix> tensor;

}; // class EigenTensor

} // namespace ocl


#endif  // OCLCPP_OCL_EIGENTENSOR_H_
