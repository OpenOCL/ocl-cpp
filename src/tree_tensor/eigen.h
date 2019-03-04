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
#ifndef OCL_EIGEN_H_
#define OCL_EIGEN_H_

#include <Eigen/Dense>

namespace ocl {

typedef float Scalar;
typedef int Integer;
typedef Eigen::MatrixXf EigenMatrixX;
typedef Eigen::Vector3f EigenVector3;
typedef Eigen::VectorXf EigenVectorX;

namespace eigen {

static inline EigenMatrixX fromScalar(Scalar s) {
  EigenMatrixX m;
  m << s;
  return m;
}

static inline EigenMatrixX Zero(Eigen::Index rows, Eigen::Index cols) {
  return EigenMatrixX::Zero(rows,cols);
}

static inline EigenMatrixX uplus(const EigenMatrixX& m) { return m; }
static inline EigenMatrixX uminus(const EigenMatrixX& m) { return -m; }
static inline EigenMatrixX square(const EigenMatrixX& m) { return Eigen::square(m.array()); }
static inline EigenMatrixX inverse(const EigenMatrixX& m) { return Eigen::inverse(m.array()); }
static inline EigenMatrixX abs(const EigenMatrixX& m) { return Eigen::abs(m.array()); }
static inline EigenMatrixX sqrt(const EigenMatrixX& m) { return Eigen::sqrt(m.array()); }
static inline EigenMatrixX sin(const EigenMatrixX& m) { return Eigen::sin(m.array()); }
static inline EigenMatrixX cos(const EigenMatrixX& m) { return Eigen::cos(m.array()); }
static inline EigenMatrixX tan(const EigenMatrixX& m) { return Eigen::tan(m.array()); }
static inline EigenMatrixX atan(const EigenMatrixX& m) { return Eigen::atan(m.array()); }
static inline EigenMatrixX asin(const EigenMatrixX& m) { return Eigen::asin(m.array()); }
static inline EigenMatrixX acos(const EigenMatrixX& m) { return Eigen::acos(m.array()); }
static inline EigenMatrixX tanh(const EigenMatrixX& m) { return Eigen::tanh(m.array()); }
static inline EigenMatrixX sinh(const EigenMatrixX& m) { return Eigen::sinh(m.array()); }
static inline EigenMatrixX cosh(const EigenMatrixX& m) { return Eigen::cosh(m.array()); }
static inline EigenMatrixX exp(const EigenMatrixX& m) { return Eigen::exp(m.array()); }
static inline EigenMatrixX log(const EigenMatrixX& m) { return Eigen::log(m.array()); }

// unary element wise + constant
static inline EigenMatrixX pow(const EigenMatrixX& m, const Scalar exponent) {
  return m.array().pow(exponent);
}

// reduction
static inline EigenMatrixX norm(const EigenMatrixX& m) { return fromScalar(m.norm()); }
static inline EigenMatrixX sum(const EigenMatrixX& m) { return fromScalar(m.sum()); }
static inline EigenMatrixX min(const EigenMatrixX& m) { return fromScalar(m.minCoeff()); }
static inline EigenMatrixX max(const EigenMatrixX& m) { return fromScalar(m.maxCoeff()); }
static inline EigenMatrixX mean(const EigenMatrixX& m) { return fromScalar(m.mean()); }
static inline EigenMatrixX trace(const EigenMatrixX& m) { return fromScalar(m.trace()); }
static inline EigenMatrixX prod(const EigenMatrixX& m) { return fromScalar(m.prod()); }

// geometrical
static inline EigenMatrixX reshape(const EigenMatrixX& m, Integer rows, Integer cols) {
  // create copy of matrix data, and create map with new dimensions but same data
  Scalar data_copy[m.size()];
  std::copy(m.data(), m.data() + m.size(), data_copy);
  Eigen::Map<EigenMatrixX> res(data_copy, rows, cols);
  return res;
}
static inline EigenMatrixX transpose(const EigenMatrixX& m) { return m.transpose(); }
// get block slice of cols (i:j) and rows (k:l)
static inline EigenMatrixX block(const EigenMatrixX& m, Integer i, Integer j, Integer k, Integer l) { return m.block(i,k,j-i+1,l-k+1); }
// get element at (i,k)
static inline EigenMatrixX slice(const EigenMatrixX& m, Integer i, Integer k) { return m.block(i,k,1,1); }

// binary coefficient wise
static inline EigenMatrixX ctimes(const EigenMatrixX& m1, const EigenMatrixX& m2) { return m1.array() * m2.array(); }
static inline EigenMatrixX cplus(const EigenMatrixX& m1, const EigenMatrixX& m2) { return m1.array() + m2.array(); }
static inline EigenMatrixX cdiv(const EigenMatrixX& m1, const EigenMatrixX& m2) { return m1.array() / m2.array(); }
static inline EigenMatrixX cminus(const EigenMatrixX& m1, const EigenMatrixX& m2) { return m1.array() - m2.array(); }

// binary operations
static inline EigenMatrixX times(const EigenMatrixX& m1, const EigenMatrixX& m2) { return m1 * m2; }

static inline EigenMatrixX cross(const EigenMatrixX& m1, const EigenMatrixX& m2) {
  // convert matrizes to vectors of length 3
  Scalar m1_data[m1.size()];
  std::copy(m1.data(), m1.data() + m1.size(), m1_data);

  Scalar m2_data[m2.size()];
  std::copy(m2.data(), m2.data() + m2.size(), m2_data);

  Eigen::Map<EigenVector3> v1(m1_data);
  Eigen::Map<EigenVector3> v2(m2_data);
  return v1.cross(v2);
}

static inline EigenMatrixX dot(const EigenMatrixX& m1, const EigenMatrixX& m2) {
  // convert matrizes to vectors
  Scalar m1_data[m1.size()];
  std::copy(m1.data(), m1.data() + m1.size(), m1_data);

  Scalar m2_data[m2.size()];
  std::copy(m2.data(), m2.data() + m2.size(), m2_data);

  Eigen::Map<EigenVectorX> v1(m1_data, m1.size());
  Eigen::Map<EigenVectorX> v2(m2_data, m2.size());
  return fromScalar(v1.dot(v2));
}

} // namespace eigen
} // namespace ocl

#endif // OCL_EIGEN_H_
