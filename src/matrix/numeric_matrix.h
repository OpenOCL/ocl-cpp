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

#include "eigen.h"

namespace ocl
{

class NumericMatrix
{
public:

  static NumericMatrix Zero(Eigen::Index rows, Eigen::Index cols) {
    return NumericMatrix(EigenMatrixX::Zero(rows,cols));
  }

  NumericMatrix(EigenMatrixX m) : m(m) { }

  // Member functions are defined inline below class (after static functions).
  NumericMatrix uplus();
  NumericMatrix uminus();
  NumericMatrix square();
  NumericMatrix inverse();
  NumericMatrix abs();
  NumericMatrix sqrt();
  NumericMatrix sin();
  NumericMatrix cos();
  NumericMatrix tan();
  NumericMatrix atan();
  NumericMatrix asin();
  NumericMatrix acos();
  NumericMatrix tanh();
  NumericMatrix sinh();
  NumericMatrix cosh();
  NumericMatrix exp();
  NumericMatrix log();

  NumericMatrix pow(const Scalar exponent);

  NumericMatrix norm();
  NumericMatrix sum();
  NumericMatrix min();
  NumericMatrix max();
  NumericMatrix mean();
  NumericMatrix trace();
  NumericMatrix prod();

  NumericMatrix reshape(const Integer rows, const Integer cols);
  NumericMatrix transpose();
  NumericMatrix block(const Integer i, const Integer j, const Integer k, const Integer l);
  NumericMatrix slice(const Integer i, const Integer k);

  NumericMatrix ctimes(const NumericMatrix& other);
  NumericMatrix cplus(const NumericMatrix& other);
  NumericMatrix cdiv(const NumericMatrix& other);
  NumericMatrix cminus(const NumericMatrix& other);

  NumericMatrix times(const NumericMatrix& other);
  NumericMatrix cross(const NumericMatrix& other);
  NumericMatrix dot(const NumericMatrix& other);

  EigenMatrixX m;

}; // class NumericMatrix

// Static functions
static inline NumericMatrix uplus(const NumericMatrix& m) { return NumericMatrix(eigen::uplus(m.m)); }
static inline NumericMatrix uminus(const NumericMatrix& m) { return NumericMatrix(eigen::uminus(m.m)); }
static inline NumericMatrix square(const NumericMatrix& m) { return NumericMatrix(eigen::square(m.m)); }
static inline NumericMatrix inverse(const NumericMatrix& m) { return NumericMatrix(eigen::inverse(m.m)); }
static inline NumericMatrix abs(const NumericMatrix& m) { return NumericMatrix(eigen::abs(m.m)); }
static inline NumericMatrix sqrt(const NumericMatrix& m) { return NumericMatrix(eigen::sqrt(m.m)); }
static inline NumericMatrix sin(const NumericMatrix& m) { return NumericMatrix(eigen::sin(m.m)); }
static inline NumericMatrix cos(const NumericMatrix& m) { return NumericMatrix(eigen::cos(m.m)); }
static inline NumericMatrix tan(const NumericMatrix& m) { return NumericMatrix(eigen::tan(m.m)); }
static inline NumericMatrix atan(const NumericMatrix& m) { return NumericMatrix(eigen::atan(m.m)); }
static inline NumericMatrix asin(const NumericMatrix& m) { return NumericMatrix(eigen::asin(m.m)); }
static inline NumericMatrix acos(const NumericMatrix& m) { return NumericMatrix(eigen::acos(m.m)); }
static inline NumericMatrix tanh(const NumericMatrix& m) { return NumericMatrix(eigen::tanh(m.m)); }
static inline NumericMatrix sinh(const NumericMatrix& m) { return NumericMatrix(eigen::sinh(m.m)); }
static inline NumericMatrix cosh(const NumericMatrix& m) { return NumericMatrix(eigen::cosh(m.m)); }
static inline NumericMatrix exp(const NumericMatrix& m) { return NumericMatrix(eigen::exp(m.m)); }
static inline NumericMatrix log(const NumericMatrix& m) { return NumericMatrix(eigen::log(m.m)); }

static inline NumericMatrix pow(const NumericMatrix& m, const Scalar exponent) { return NumericMatrix(eigen::pow(m.m, exponent)); }

static inline NumericMatrix norm(const NumericMatrix& m) { return NumericMatrix(eigen::norm(m.m)); }
static inline NumericMatrix sum(const NumericMatrix& m) { return NumericMatrix(eigen::sum(m.m)); }
static inline NumericMatrix min(const NumericMatrix& m) { return NumericMatrix(eigen::min(m.m)); }
static inline NumericMatrix max(const NumericMatrix& m) { return NumericMatrix(eigen::max(m.m)); }
static inline NumericMatrix mean(const NumericMatrix& m) { return NumericMatrix(eigen::mean(m.m)); }
static inline NumericMatrix trace(const NumericMatrix& m) { return NumericMatrix(eigen::trace(m.m)); }
static inline NumericMatrix prod(const NumericMatrix& m) { return NumericMatrix(eigen::prod(m.m)); }

static inline NumericMatrix reshape(const NumericMatrix& m, const Integer rows, const Integer cols) {
  return NumericMatrix(eigen::reshape(m.m, rows, cols));
}
static inline NumericMatrix transpose(const NumericMatrix& m) {
  return NumericMatrix(eigen::transpose(m.m));
}
static inline NumericMatrix block(const NumericMatrix& m, const Integer i, const Integer j, const Integer k, const Integer l) {
  return NumericMatrix(eigen::block(m.m, i, j, k, l));
}
static inline NumericMatrix slice(const NumericMatrix& m, const Integer i, const Integer k) {
  return NumericMatrix(eigen::slice(m.m, i, k));
}

static inline NumericMatrix ctimes(const NumericMatrix& m1, const NumericMatrix& m2) { return NumericMatrix(eigen::ctimes(m1.m, m2.m)); }
static inline NumericMatrix cplus(const NumericMatrix& m1, const NumericMatrix& m2) { return NumericMatrix(eigen::cplus(m1.m, m2.m)); }
static inline NumericMatrix cdiv(const NumericMatrix& m1, const NumericMatrix& m2) { return NumericMatrix(eigen::cdiv(m1.m, m2.m)); }
static inline NumericMatrix cminus(const NumericMatrix& m1, const NumericMatrix& m2) { return NumericMatrix(eigen::cminus(m1.m, m2.m)); }

static inline NumericMatrix times(const NumericMatrix& m1, const NumericMatrix& m2) { return NumericMatrix(eigen::times(m1.m, m2.m)); }
static inline NumericMatrix cross(const NumericMatrix& m1, const NumericMatrix& m2) { return NumericMatrix(eigen::cross(m1.m, m2.m)); }
static inline NumericMatrix dot(const NumericMatrix& m1, const NumericMatrix& m2) { return NumericMatrix(eigen::dot(m1.m, m2.m)); }

// Member functions (calling the static functions above)
inline NumericMatrix NumericMatrix::uplus() { return ocl::uplus(*this); }
inline NumericMatrix NumericMatrix::uminus() { return ocl::uminus(*this); }
inline NumericMatrix NumericMatrix::square() { return ocl::square(*this); }
inline NumericMatrix NumericMatrix::inverse() { return ocl::inverse(*this); }
inline NumericMatrix NumericMatrix::abs() { return ocl::abs(*this); }
inline NumericMatrix NumericMatrix::sqrt() { return ocl::sqrt(*this); }
inline NumericMatrix NumericMatrix::sin() { return ocl::sin(*this); }
inline NumericMatrix NumericMatrix::cos() { return ocl::cos(*this); }
inline NumericMatrix NumericMatrix::tan() { return ocl::tan(*this); }
inline NumericMatrix NumericMatrix::atan() { return ocl::atan(*this); }
inline NumericMatrix NumericMatrix::asin() { return ocl::asin(*this); }
inline NumericMatrix NumericMatrix::acos() { return ocl::acos(*this); }
inline NumericMatrix NumericMatrix::tanh() { return ocl::tanh(*this); }
inline NumericMatrix NumericMatrix::sinh() { return ocl::sinh(*this); }
inline NumericMatrix NumericMatrix::cosh() { return ocl::cosh(*this); }
inline NumericMatrix NumericMatrix::exp() { return ocl::exp(*this); }
inline NumericMatrix NumericMatrix::log() { return ocl::log(*this); }

inline NumericMatrix NumericMatrix::pow(const Scalar exponent) {
  return ocl::pow(*this, exponent);
}

inline NumericMatrix NumericMatrix::norm() { return ocl::norm(*this); }
inline NumericMatrix NumericMatrix::sum() { return ocl::sum(*this); }
inline NumericMatrix NumericMatrix::min() { return ocl::min(*this); }
inline NumericMatrix NumericMatrix::max() { return ocl::max(*this); }
inline NumericMatrix NumericMatrix::mean() { return ocl::mean(*this); }
inline NumericMatrix NumericMatrix::trace() { return ocl::trace(*this); }
inline NumericMatrix NumericMatrix::prod() { return ocl::prod(*this); }

inline NumericMatrix NumericMatrix::reshape(const Integer rows, const Integer cols) {
  return ocl::reshape(*this, rows, cols);
}
inline NumericMatrix NumericMatrix::transpose() { return ocl::transpose(*this); }
inline NumericMatrix NumericMatrix::block(const Integer i, const Integer j, const Integer k, const Integer l) {
  return ocl::block(*this, i, j, k, l);
}
inline NumericMatrix NumericMatrix::slice(const Integer i, const Integer k) {
  return ocl::slice(*this, i, k);
}

inline NumericMatrix NumericMatrix::ctimes(const NumericMatrix& other) { return ocl::ctimes(*this, other); }
inline NumericMatrix NumericMatrix::cplus(const NumericMatrix& other) { return ocl::cplus(*this, other); }
inline NumericMatrix NumericMatrix::cdiv(const NumericMatrix& other) { return ocl::cdiv(*this, other); }
inline NumericMatrix NumericMatrix::cminus(const NumericMatrix& other) { return ocl::cminus(*this, other); }

inline NumericMatrix NumericMatrix::times(const NumericMatrix& other) { return ocl::times(*this, other); }
inline NumericMatrix NumericMatrix::cross(const NumericMatrix& other) { return ocl::cross(*this, other); }
inline NumericMatrix NumericMatrix::dot(const NumericMatrix& other) { return ocl::dot(*this, other); }

} // namespace ocl
#endif // OCLCPP_OCL_EIGENMATRIX_H_
