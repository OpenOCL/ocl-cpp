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
#ifndef OCLCPP_OCL_EIGENMATRIX_H_
#define OCLCPP_OCL_EIGENMATRIX_H_

namespace ocl
{

class NumericMatrix
{
  typedef Eigen::MatrixXf EigenMatrixX;
  typedef Eigen::Vector3f EigenVector3;
  typedef Eigen::VectorXf EigenVectorX;

  // Function pointers to static functions
  typedef EigenMatrixX (*UnaryOpFcn)(const EigenMatrixX& m);
  typedef EigenMatrixX (*UnaryOpFcnWithScalar)(const EigenMatrixX& m, Scalar s);
  typedef EigenMatrixX (*UnaryOpFcnWithInteger2)(const EigenMatrixX& m, Integer s1, Integer s2);
  typedef EigenMatrixX (*UnaryOpFcnWithInteger4)(const EigenMatrixX& m, Integer s1, Integer s2, Integer s3, Integer s4);
  typedef Scalar (*UnaryReductionOpFcn)(const EigenMatrixX& m);

  typedef EigenMatrixX (*BinaryOpFcn)(const EigenMatrixX& m1, const EigenMatrixX& m2);

  static EigenMatrixX fromScalar(Scalar s) {
    EigenMatrix m;
    m << s;
    return m;
  }

  static EigenMatrixX m_uplus(const EigenMatrixX& m) { return m; }
  static EigenMatrixX m_uminus(const EigenMatrixX& m) { return -m; }
  static EigenMatrixX m_square(const EigenMatrixX& m) { return Eigen::square(m.array()); }
  static EigenMatrixX m_inverse(const EigenMatrixX& m) { return Eigen::inverse(m.array()); }
  static EigenMatrixX m_abs(const EigenMatrixX& m) { return Eigen::abs(m.array()); }
  static EigenMatrixX m_sqrt(const EigenMatrixX& m) { return Eigen::sqrt(m.array()); }
  static EigenMatrixX m_sin(const EigenMatrixX& m) { return Eigen::sin(m.array()); }
  static EigenMatrixX m_cos(const EigenMatrixX& m) { return Eigen::cos(m.array()); }
  static EigenMatrixX m_tan(const EigenMatrixX& m) { return Eigen::tan(m.array()); }
  static EigenMatrixX m_atan(const EigenMatrixX& m) { return Eigen::atan(m.array()); }
  static EigenMatrixX m_asin(const EigenMatrixX& m) { return Eigen::asin(m.array()); }
  static EigenMatrixX m_acos(const EigenMatrixX& m) { return Eigen::acos(m.array()); }
  static EigenMatrixX m_tanh(const EigenMatrixX& m) { return Eigen::tanh(m.array()); }
  static EigenMatrixX m_sinh(const EigenMatrixX& m) { return Eigen::sinh(m.array()); }
  static EigenMatrixX m_cosh(const EigenMatrixX& m) { return Eigen::cosh(m.array()); }
  static EigenMatrixX m_exp(const EigenMatrixX& m) { return Eigen::exp(m.array()); }
  static EigenMatrixX m_log(const EigenMatrixX& m) { return Eigen::log(m.array()); }

  // unary element wise + constant
  static EigenMatrixX m_pow(const EigenMatrix& m, const Scalar exponent) { return m.array().pow(exponent); }

  // reduction
  static EigenMatrixX m_norm(const EigenMatrix& m) { return fromScalar(m.norm()); }
  static EigenMatrixX m_sum(const EigenMatrix& m) { return fromScalar(m.sum()); }
  static EigenMatrixX m_min(const EigenMatrix& m) { return fromScalar(m.minCoeff()); }
  static EigenMatrixX m_max(const EigenMatrix& m) { return fromScalar(m.maxCoeff()); }
  static EigenMatrixX m_mean(const EigenMatrix& m) { return fromScalar(m.mean()); }
  static EigenMatrixX m_trace(const EigenMatrix& m) { return fromScalar(m.trace()); }
  static EigenMatrixX m_prod(const EigenMatrix& m) { return fromScalar(m.prod()); }

  // geometrical
  static EigenMatrixX m_reshape(const EigenMatrixX& m, Integer rows, Integer cols) {
    // create copy of matrix data, and create map with new dimensions but same data
    Scalar data_copy[m.size()];
    std::copy(m.data(), m.data() + m.size(), data_copy);
    Eigen::Map<EigenMatrixX> res(data_copy, rows, cols);
    return res;
  }
  static EigenMatrixX m_transpose(const EigenMatrixX& m) { return m.transpose(); }
  // get block slice of cols (i:j) and rows (k:l)
  static EigenMatrixX m_block(const EigenMatrixX& m, Integer i, Integer j, Integer k, Integer l) { return m.block(i,k,j-i+1,l-k+1); }
  // get element at (i,k)
  static EigenMatrixX m_slice(const EigenMatrixX& m, Integer i, Integer k) { return m.block(i,k,1,1); }

  // binary coefficient wise
  static EigenMatrixX m_ctimes(const EigenMatrixX& m1, const EigenMatrixX& m2) { return m1.array() * m2.array(); }
  static EigenMatrixX m_cplus(const EigenMatrixX& m1, const EigenMatrixX& m2) { return m1.array() + m2.array(); }
  static EigenMatrixX m_cdiv(const EigenMatrixX& m1, const EigenMatrixX& m2) { return m1.array() / m2.array(); }
  static EigenMatrixX m_cminus(const EigenMatrixX& m1, const EigenMatrixX& m2) { return m1.array() - m2.array(); }

  // binary operations
  static EigenMatrixX m_times(const EigenMatrixX& m1, const EigenMatrixX& m2) { return m1 * m2; }

  static EigenMatrixX m_cross(const EigenMatrixX& m1, const EigenMatrixX& m2) {
    // convert matrizes to vectors of length 3
    Scalar m1_data[m1.size()];
    std::copy(m1.data(), m1.data() + m1.size(), m1_data);

    Scalar m2_data[m2.size()];
    std::copy(m2.data(), m2.data() + m2.size(), m2_data);

    Eigen::Map<EigenVector3> v1(m1_data);
    Eigen::Map<EigenVector3> v2(m2_data);
    return v1.cross(v2);
  }

  static EigenMatrixX m_dot(const EigenMatrixX& m1, const EigenMatrixX& m2) {
    // convert matrizes to vectors
    Scalar m1_data[m1.size()];
    std::copy(m1.data(), m1.data() + m1.size(), m1_data);

    Scalar m2_data[m2.size()];
    std::copy(m2.data(), m2.data() + m2.size(), m2_data);

    Eigen::Map<EigenVector> v1(m1_data, m1.size());
    Eigen::Map<EigenVector> v2(m2_data, m2.size());
    return fromScalar(v1.dot(v2));
  }




};

}
#endif // OCLCPP_OCL_EIGENMATRIX_H_
