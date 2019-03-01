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
class EigenTensor
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
  EigenTensor(Eigen::Index rows, Eigen::Index cols) : tensor(1, EigenMatrix::Zero(rows,cols)) { }
  EigenTensor() { }

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
  EigenTensor unaryVecOperation(UnaryOpFcn fcn_ptr) const
  {
    EigenTensor t = EigenTensor();
    for(unsigned int i=0; i<tensor.size(); i++) {
      t.tensor.push_back( fcn_ptr(tensor[i]) );
    }
    return t;
  }

  EigenTensor unaryVecOperationWithScalar(UnaryOpFcnWithScalar fcn_ptr, Scalar s) const
  {
    EigenTensor t = EigenTensor();
    for(unsigned int i=0; i<tensor.size(); i++) {
      t.tensor.push_back( fcn_ptr(tensor[i], s) );
    }
    return t;
  }

  EigenTensor unaryVecOperationWithInteger2(UnaryOpFcnWithInteger2 fcn_ptr, Integer s1, Integer s2) const
  {
    EigenTensor t = EigenTensor();
    for(unsigned int i=0; i<tensor.size(); i++) {
      t.tensor.push_back( fcn_ptr(tensor[i], s1, s2) );
    }
    return t;
  }

  Tensor unaryVecOperationWithInteger4(UnaryOpFcnWithInteger4 fcn_ptr, Integer s1, Integer s2, Integer s3, Integer s4) const
  {
    EigenTensor t = EigenTensor();
    for(unsigned int i=0; i<tensor.size(); i++) {
      t.tensor.push_back( fcn_ptr(tensor[i], s1, s2, s3, s4) );
    }
    return t;
  }

  EigenTensor unaryReductionOperation(UnaryReductionOpFcn fcn_ptr) const
  {
    EigenTensor t = Tensor();
    for(unsigned int i=0; i<tensor.size(); i++) {
      Scalar s = fcn_ptr(tensor[i]);
      EigenMatrix m;
      m << s;
      t.tensor.push_back( m );
    }
    return t;
  }

  EigenTensor binaryVecOperation(BinaryOpFcn fcn_ptr, const Tensor& other) const
  {
    // TODO: implement broadcasting
    //assertEqual(other.tensor.size(), 1);

    EigenTensor t = EigenTensor();
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
  EigenTensor uplus() const { return unaryVecOperation(&EigenTensor::m_uplus); }
  EigenTensor uminus() const { return unaryVecOperation(&EigenTensor::m_uminus); }
  EigenTensor square() const { return unaryVecOperation(&EigenTensor::m_square); }
  EigenTensor inverse() const { return unaryVecOperation(&EigenTensor::m_inverse); }
  EigenTensor abs() const { return unaryVecOperation(&EigenTensor::m_abs); }
  EigenTensor sqrt() const { return unaryVecOperation(&EigenTensor::m_sqrt); }
  EigenTensor sin() const { return unaryVecOperation(&EigenTensor::m_sin); }
  EigenTensor cos() const { return unaryVecOperation(&EigenTensor::m_cos); }
  EigenTensor tan() const { return unaryVecOperation(&EigenTensor::m_tan); }
  EigenTensor atan() const { return unaryVecOperation(&EigenTensor::m_atan); }
  EigenTensor asin() const { return unaryVecOperation(&EigenTensor::m_asin); }
  EigenTensor acos() const { return unaryVecOperation(&EigenTensor::m_acos); }
  EigenTensor tanh() const { return unaryVecOperation(&EigenTensor::m_tanh); }
  EigenTensor cosh() const { return unaryVecOperation(&EigenTensor::m_cosh); }
  EigenTensor sinh() const { return unaryVecOperation(&EigenTensor::m_sinh); }
  EigenTensor exp() const { return unaryVecOperation(&EigenTensor::m_exp); }
  EigenTensor log() const { return unaryVecOperation(&EigenTensor::m_log); }

  // operators - unary element wise + scalar
  EigenTensor pow(Scalar exponent) const {
    return unaryVecOperationWithScalar(&EigenTensor::m_pow, exponent);
  }

  // reduction operations
  EigenTensor norm() const { return unaryReductionOperation(&EigenTensor::m_norm); }
  EigenTensor sum() const { return unaryReductionOperation(&EigenTensor::m_sum); }
  EigenTensor min() const { return unaryReductionOperation(&EigenTensor::m_min); }
  EigenTensor max() const { return unaryReductionOperation(&EigenTensor::m_max); }
  EigenTensor trace() const { return unaryReductionOperation(&EigenTensor::m_trace); }
  EigenTensor mean() const { return unaryReductionOperation(&EigenTensor::m_mean); }
  EigenTensor prod() const { return unaryReductionOperation(&EigenTensor::m_prod); }

  // geometrical operations
  EigenTensor transpose() const { return unaryVecOperation(&EigenTensor::m_transpose); }

  EigenTensor reshape(Integer cols, Integer rows) const {
    return unaryVecOperationWithInteger2(&EigenTensor::m_reshape, cols, rows);
  }

  // get slice (i:j)
  EigenTensor slice(Integer i, Integer j) const {
    return unaryVecOperationWithInteger2(&EigenTensor::m_slice, i, j);
  }

  // get block slice of cols (i:j) and rows (k:l)
  EigenTensor slice(Integer i, Integer j, Integer k, Integer l) const {
    return unaryVecOperationWithInteger4(&EigenTensor::m_block, i, j, k, l);
  }

  // binary coefficient wise
  EigenTensor plus(const Tensor& other) const { return binaryVecOperation(&EigenTensor::m_cplus, other); }
  EigenTensor minus(const Tensor& other) const { return binaryVecOperation(&EigenTensor::m_cminus, other); }
  EigenTensor ctimes(const Tensor& other) const { return binaryVecOperation(&EigenTensor::m_ctimes, other); }
  EigenTensor cdivide(const Tensor& other) const { return binaryVecOperation(&EigenTensor::m_cdiv, other); }

  // binary matrix operations
  EigenTensor times(const Tensor& other) const { return binaryVecOperation(&EigenTensor::m_times, other); }
  EigenTensor cross(const Tensor& other) const { return binaryVecOperation(&EigenTensor::m_cross, other); }
  EigenTensor dot(const Tensor& other) const { return binaryVecOperation(&EigenTensor::m_dot, other); }

  // operator overloading
  EigenTensor operator+(const Tensor& other) {
    return this->plus(other);
  }
  EigenTensor operator-(const Tensor& other) {
    return this->minus(other);
  }
  EigenTensor operator*(const Tensor& other) {
    return this->times(other);
  }
  EigenTensor operator/(const Tensor& other) {
    return this->cdivide(other);
  }

 private:
  std::vector<EigenMatrix> tensor;

}; // class EigenTensor

} // namespace ocl


#endif  // OCLCPP_OCL_EIGENTENSOR_H_
