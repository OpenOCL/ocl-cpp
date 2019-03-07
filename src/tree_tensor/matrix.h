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
#ifndef OCLCPP_OCL_SYMBOLIC_AD_MATRIX_H_
#define OCLCPP_OCL_SYMBOLIC_AD_MATRIX_H_

#include "casadi/casadi.hpp"
#include "casadi.h"

#include "typedefs.h"

namespace ocl
{


class Matrix
{
public:

  static Matrix Sym(int rows, int cols) {
    CasadiMatrixNat m = CasadiMatrixNat::sym("m", rows, cols);
    return Matrix(m);
  }

  Matrix(const ::casadi::DM& v) : m(v) { }
  Matrix(const double v) : m(v) { }
  Matrix(const CasadiMatrixNat& m) : m(m) { }

  CasadiMatrixNat data() const{
    return m;
  }

  // Member functions are defined inline below class (after static functions).
  Matrix uplus() const;
  Matrix uminus() const;
  Matrix square() const;
  Matrix inverse() const;
  Matrix abs() const;
  Matrix sqrt() const;
  Matrix sin() const;
  Matrix cos() const;
  Matrix tan() const;
  Matrix atan() const;
  Matrix asin() const;
  Matrix acos() const;
  Matrix tanh() const;
  Matrix sinh() const;
  Matrix cosh() const;
  Matrix exp() const;
  Matrix log() const;

  Matrix pow(const Scalar exponent) const;

  Matrix norm() const;
  Matrix sum() const;
  Matrix min() const;
  Matrix max() const;
  Matrix mean() const;
  Matrix trace() const;

  Matrix reshape(const Integer rows, const Integer cols) const;
  Matrix transpose() const;
  Matrix block(const Integer i, const Integer j, const Integer k, const Integer l) const;
  Matrix slice(const Integer i, const Integer k) const;

  Matrix ctimes(const Matrix& other) const;
  Matrix plus(const Matrix& other) const;
  Matrix cdiv(const Matrix& other) const;
  Matrix minus(const Matrix& other) const;

  Matrix times(const Matrix& other) const;
  Matrix cross(const Matrix& other) const;
  Matrix dot(const Matrix& other) const;

private:
  CasadiMatrixNat m;
};

static inline Shape shape(const Matrix& m) {
  return Shape({m.data().rows(), m.data().columns()});
}

static inline std::vector<double> full(const Matrix& m) {
  double *data = m.data().ptr();
  int nel = m.shape().numel();
  vector<double> values(data, data + nel);
  return value;
}

// Static functions
static inline Matrix uplus(const Matrix& m) { return Matrix(casadi::uplus(m.data())); }
static inline Matrix uminus(const Matrix& m) { return Matrix(casadi::uminus(m.data())); }
static inline Matrix square(const Matrix& m) { return Matrix(casadi::square(m.data())); }
static inline Matrix inverse(const Matrix& m) { return Matrix(casadi::inverse(m.data())); }
static inline Matrix abs(const Matrix& m) { return Matrix(casadi::abs(m.data())); }
static inline Matrix sqrt(const Matrix& m) { return Matrix(casadi::sqrt(m.data())); }
static inline Matrix sin(const Matrix& m) { return Matrix(casadi::sin(m.data())); }
static inline Matrix cos(const Matrix& m) { return Matrix(casadi::cos(m.data())); }
static inline Matrix tan(const Matrix& m) { return Matrix(casadi::tan(m.data())); }
static inline Matrix atan(const Matrix& m) { return Matrix(casadi::atan(m.data())); }
static inline Matrix asin(const Matrix& m) { return Matrix(casadi::asin(m.data())); }
static inline Matrix acos(const Matrix& m) { return Matrix(casadi::acos(m.data())); }
static inline Matrix tanh(const Matrix& m) { return Matrix(casadi::tanh(m.data())); }
static inline Matrix sinh(const Matrix& m) { return Matrix(casadi::sinh(m.data())); }
static inline Matrix cosh(const Matrix& m) { return Matrix(casadi::cosh(m.data())); }
static inline Matrix exp(const Matrix& m) { return Matrix(casadi::exp(m.data())); }
static inline Matrix log(const Matrix& m) { return Matrix(casadi::log(m.data())); }

static inline Matrix pow(const Matrix& m, const Scalar exponent) { return Matrix(casadi::pow(m.data(), exponent)); }

static inline Matrix norm(const Matrix& m) { return Matrix(casadi::norm(m.data())); }
static inline Matrix sum(const Matrix& m) { return Matrix(casadi::sum(m.data())); }
static inline Matrix min(const Matrix& m) { return Matrix(casadi::min(m.data())); }
static inline Matrix max(const Matrix& m) { return Matrix(casadi::max(m.data())); }
static inline Matrix mean(const Matrix& m) { return Matrix(casadi::mean(m.data())); }
static inline Matrix trace(const Matrix& m) { return Matrix(casadi::trace(m.data())); }

static inline Matrix reshape(const Matrix& m, const Integer rows, const Integer cols) {
  return Matrix(casadi::reshape(m.data(), rows, cols));
}
static inline Matrix transpose(const Matrix& m) {
  return Matrix(casadi::transpose(m.data()));
}
static inline Matrix block(const Matrix& m, const Integer i, const Integer j, const Integer k, const Integer l) {
  return Matrix(casadi::block(m.data(), i, j, k, l));
}
static inline Matrix slice(const Matrix& m, const Integer i, const Integer k) {
  return Matrix(casadi::slice(m.data(), i, k));
}

static inline Matrix ctimes(const Matrix& m1, const Matrix& m2) { return Matrix(casadi::ctimes(m1.data(), m2.data())); }
static inline Matrix plus(const Matrix& m1, const Matrix& m2) { return Matrix(casadi::plus(m1.data(), m2.data())); }
static inline Matrix cdiv(const Matrix& m1, const Matrix& m2) { return Matrix(casadi::cdiv(m1.data(), m2.data())); }
static inline Matrix minus(const Matrix& m1, const Matrix& m2) { return Matrix(casadi::minus(m1.data(), m2.data())); }

static inline Matrix times(const Matrix& m1, const Matrix& m2) { return Matrix(casadi::times(m1.data(), m2.data())); }
static inline Matrix cross(const Matrix& m1, const Matrix& m2) { return Matrix(casadi::cross(m1.data(), m2.data())); }
static inline Matrix dot(const Matrix& m1, const Matrix& m2) { return Matrix(casadi::dot(m1.data(), m2.data())); }

// Member functions (calling the static functions above)
inline Matrix Matrix::uplus() { return ocl::uplus(*this); }
inline Matrix Matrix::uminus() { return ocl::uminus(*this); }
inline Matrix Matrix::square() { return ocl::square(*this); }
inline Matrix Matrix::inverse() { return ocl::inverse(*this); }
inline Matrix Matrix::abs() { return ocl::abs(*this); }
inline Matrix Matrix::sqrt() { return ocl::sqrt(*this); }
inline Matrix Matrix::sin() { return ocl::sin(*this); }
inline Matrix Matrix::cos() { return ocl::cos(*this); }
inline Matrix Matrix::tan() { return ocl::tan(*this); }
inline Matrix Matrix::atan() { return ocl::atan(*this); }
inline Matrix Matrix::asin() { return ocl::asin(*this); }
inline Matrix Matrix::acos() { return ocl::acos(*this); }
inline Matrix Matrix::tanh() { return ocl::tanh(*this); }
inline Matrix Matrix::sinh() { return ocl::sinh(*this); }
inline Matrix Matrix::cosh() { return ocl::cosh(*this); }
inline Matrix Matrix::exp() { return ocl::exp(*this); }
inline Matrix Matrix::log() { return ocl::log(*this); }

inline Matrix Matrix::pow(const Scalar exponent) {
  return ocl::pow(*this, exponent);
}

inline Matrix Matrix::norm() { return ocl::norm(*this); }
inline Matrix Matrix::sum() { return ocl::sum(*this); }
inline Matrix Matrix::min() { return ocl::min(*this); }
inline Matrix Matrix::max() { return ocl::max(*this); }
inline Matrix Matrix::mean() { return ocl::mean(*this); }
inline Matrix Matrix::trace() { return ocl::trace(*this); }

inline Matrix Matrix::reshape(const Integer rows, const Integer cols) {
  return ocl::reshape(*this, rows, cols);
}
inline Matrix Matrix::transpose() { return ocl::transpose(*this); }
inline Matrix Matrix::block(const Integer i, const Integer j, const Integer k, const Integer l) {
  return ocl::block(*this, i, j, k, l);
}
inline Matrix Matrix::slice(const Integer i, const Integer k) {
  return ocl::slice(*this, i, k);
}

inline Matrix Matrix::ctimes(const Matrix& other) { return ocl::ctimes(*this, other); }
inline Matrix Matrix::plus(const Matrix& other) { return ocl::plus(*this, other); }
inline Matrix Matrix::cdiv(const Matrix& other) { return ocl::cdiv(*this, other); }
inline Matrix Matrix::minus(const Matrix& other) { return ocl::minus(*this, other); }

inline Matrix Matrix::times(const Matrix& other) { return ocl::times(*this, other); }
inline Matrix Matrix::cross(const Matrix& other) { return ocl::cross(*this, other); }
inline Matrix Matrix::dot(const Matrix& other) { return ocl::dot(*this, other); }

}
#endif // OCLCPP_OCL_SYMBOLIC_AD_MATRIX_H_
