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

#ifndef OCLCPP_OCL_MATRIX_H_
#define OCLCPP_OCL_MATRIX_H_

#include "casadi.h"
#include "utils/typedefs.h"
#include "utils/slicing.h"

// File summary:
//  Defines class ocl::Matrix.
//  Static operations on ocl::Matrix.

namespace ocl
{

class Matrix : public Slicable
{
public:

  static Matrix Sym(const int rows, const int cols) {
    return Matrix(casadi::Sym(rows, cols));
  }

  static Matrix Eye(const int n) {
    return Matrix(casadi::Eye(n));
  }

  static Matrix Zero(const int rows, const int cols) {
    return Matrix(casadi::Zero(rows, cols));
  }

  static Matrix One(const int rows, const int cols) {
    return Matrix(casadi::One(rows, cols));
  }

  Matrix(const ::casadi::DM& v) : m(v) { }
  Matrix(const double v) : m(v) { }
  Matrix(const std::vector<double>& v) : m(v) { }
  Matrix(const CasadiMatrix& m) : m(m) { }

  // Get underlying data type
  CasadiMatrix raw() const { return m; }
  CasadiMatrix& rawRef() { return m; }

  CasadiMatrix data() const { return casadi::reshape(m, size(0)*size(1), 1); }

  virtual int size(const int dim) const override {
    return casadi::size(m, dim);
  }

  std::vector<double> full(const std::vector<Matrix>& variables, const std::vector<Matrix>& values) const;

  // Member functions are defined inline below class (after static functions).
  void assign(int row, int col, double val);
  void assign(const std::vector<int>& rows, int col, const Matrix& values);

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

  Matrix cpow(const Matrix& exponent) const;

  Matrix norm() const;
  Matrix sum() const;
  Matrix min() const;
  Matrix max() const;
  Matrix mean() const;
  Matrix trace() const;

  Matrix reshape(const Integer rows, const Integer cols) const;
  Matrix transpose() const;
  Matrix slice(const std::vector<int>& slice1, const std::vector<int>& slice2) const;

  Matrix ctimes(const Matrix& other) const;
  Matrix plus(const Matrix& other) const;
  Matrix cdivide(const Matrix& other) const;
  Matrix minus(const Matrix& other) const;

  Matrix cmin(const Matrix& other) const;
  Matrix cmax(const Matrix& other) const;

  Matrix times(const Matrix& other) const;
  Matrix cross(const Matrix& other) const;
  Matrix dot(const Matrix& other) const;

  Matrix atan2(const Matrix& other) const;

private:
  CasadiMatrix m;
};

static inline Matrix vertcat(const Matrix& m1, const Matrix& m2)
{
  return vertcat(m1.raw(),m2.raw());
}

static inline CasadiMatrix raw(const Matrix& m) { return m.raw(); }

static inline std::vector<int> shape(const Matrix& m) {
  return casadi::shape(m.raw());
}

static inline std::vector<double> full(const Matrix& m,
                                       const std::vector<Matrix>& variables = {},
                                       const std::vector<Matrix>& values = {})
{
  // convert vectors of Matrix into vectors of CasadiMatrix (casadi::SX)
  std::vector<CasadiMatrix> casadi_variables(variables.size());
  std::vector<CasadiMatrix> casadi_values(values.size());
  std::transform (variables.begin(), variables.end(), casadi_variables.begin(), raw);
  std::transform (values.begin(), values.end(), casadi_values.begin(), raw);

  return casadi::full(m.raw(), casadi_variables, casadi_values);
}

// Static functions
static inline void assign(Matrix& m, int row, int col, double val) {
  casadi::assign(m.rawRef(), row, col, val);
}

static inline void assign(Matrix& m, const std::vector<int>& rows, int col,
                          const Matrix& v) {
  casadi::assign(m.rawRef(), rows, col, v.raw());
}

static inline void assign(Matrix& m, const std::vector<int>& rows, int col,
                          const CasadiMatrix& values) {
  casadi::assign(m.rawRef(), rows, col, values);
}

static inline Matrix uplus(const Matrix& m) { return Matrix(casadi::uplus(m.raw())); }
static inline Matrix uminus(const Matrix& m) { return Matrix(casadi::uminus(m.raw())); }
static inline Matrix square(const Matrix& m) { return Matrix(casadi::square(m.raw())); }
static inline Matrix inverse(const Matrix& m) { return Matrix(casadi::inverse(m.raw())); }
static inline Matrix abs(const Matrix& m) { return Matrix(casadi::abs(m.raw())); }
static inline Matrix sqrt(const Matrix& m) { return Matrix(casadi::sqrt(m.raw())); }
static inline Matrix sin(const Matrix& m) { return Matrix(casadi::sin(m.raw())); }
static inline Matrix cos(const Matrix& m) { return Matrix(casadi::cos(m.raw())); }
static inline Matrix tan(const Matrix& m) { return Matrix(casadi::tan(m.raw())); }
static inline Matrix atan(const Matrix& m) { return Matrix(casadi::atan(m.raw())); }
static inline Matrix asin(const Matrix& m) { return Matrix(casadi::asin(m.raw())); }
static inline Matrix acos(const Matrix& m) { return Matrix(casadi::acos(m.raw())); }
static inline Matrix tanh(const Matrix& m) { return Matrix(casadi::tanh(m.raw())); }
static inline Matrix sinh(const Matrix& m) { return Matrix(casadi::sinh(m.raw())); }
static inline Matrix cosh(const Matrix& m) { return Matrix(casadi::cosh(m.raw())); }
static inline Matrix exp(const Matrix& m) { return Matrix(casadi::exp(m.raw())); }
static inline Matrix log(const Matrix& m) { return Matrix(casadi::log(m.raw())); }

static inline Matrix cpow(const Matrix& m, const Matrix& exponent) { return Matrix(casadi::cpow(m.raw(), exponent.raw())); }

static inline Matrix norm(const Matrix& m) { return Matrix(casadi::norm(m.raw())); }
static inline Matrix sum(const Matrix& m) { return Matrix(casadi::sum(m.raw())); }
static inline Matrix min(const Matrix& m) { return Matrix(casadi::min(m.raw())); }
static inline Matrix max(const Matrix& m) { return Matrix(casadi::max(m.raw())); }
static inline Matrix mean(const Matrix& m) { return Matrix(casadi::mean(m.raw())); }
static inline Matrix trace(const Matrix& m) { return Matrix(casadi::trace(m.raw())); }

static inline Matrix reshape(const Matrix& m, const Integer rows, const Integer cols) {
  return Matrix(casadi::reshape(m.raw(), rows, cols));
}
static inline Matrix transpose(const Matrix& m) {
  return Matrix(casadi::transpose(m.raw()));
}

static inline Matrix slice(const Matrix& m, const std::vector<int>& slice1, const std::vector<int>& slice2) {
  return Matrix(casadi::slice(m.raw(), slice1, slice2));
}

static inline Matrix ctimes(const Matrix& m1, const Matrix& m2) { return Matrix(casadi::ctimes(m1.raw(), m2.raw())); }
static inline Matrix plus(const Matrix& m1, const Matrix& m2) { return Matrix(casadi::plus(m1.raw(), m2.raw())); }
static inline Matrix cdivide(const Matrix& m1, const Matrix& m2) { return Matrix(casadi::cdivide(m1.raw(), m2.raw())); }
static inline Matrix minus(const Matrix& m1, const Matrix& m2) { return Matrix(casadi::minus(m1.raw(), m2.raw())); }

static inline Matrix cmin(const Matrix& m1, const Matrix& m2) { return Matrix(casadi::cmin(m1.raw(), m2.raw())); }
static inline Matrix cmax(const Matrix& m1, const Matrix& m2) { return Matrix(casadi::cmax(m1.raw(), m2.raw())); }

static inline Matrix times(const Matrix& m1, const Matrix& m2) { return Matrix(casadi::times(m1.raw(), m2.raw())); }
static inline Matrix cross(const Matrix& m1, const Matrix& m2) { return Matrix(casadi::cross(m1.raw(), m2.raw())); }
static inline Matrix dot(const Matrix& m1, const Matrix& m2) { return Matrix(casadi::dot(m1.raw(), m2.raw())); }

static inline Matrix atan2(const Matrix& m1, const Matrix& m2) { return Matrix(casadi::atan2(m1.raw(), m2.raw())); }

// Member functions (calling the static functions above)
inline std::vector<double> Matrix::full(const std::vector<Matrix>& variables = {}, const std::vector<Matrix>& values = {}) const { return ocl::full(*this, variables, values); }

inline void Matrix::assign(int row, int col, double val) { return ocl::assign(*this, row, col, val); }
inline void Matrix::assign(const std::vector<int>& rows, int col, const Matrix& values) { return ocl::assign(*this, rows, col, values); }

inline Matrix Matrix::uplus() const { return ocl::uplus(*this); }
inline Matrix Matrix::uminus() const { return ocl::uminus(*this); }
inline Matrix Matrix::square() const { return ocl::square(*this); }
inline Matrix Matrix::inverse() const { return ocl::inverse(*this); }
inline Matrix Matrix::abs() const { return ocl::abs(*this); }
inline Matrix Matrix::sqrt() const { return ocl::sqrt(*this); }
inline Matrix Matrix::sin() const { return ocl::sin(*this); }
inline Matrix Matrix::cos() const { return ocl::cos(*this); }
inline Matrix Matrix::tan() const { return ocl::tan(*this); }
inline Matrix Matrix::atan() const { return ocl::atan(*this); }
inline Matrix Matrix::asin() const { return ocl::asin(*this); }
inline Matrix Matrix::acos() const { return ocl::acos(*this); }
inline Matrix Matrix::tanh() const { return ocl::tanh(*this); }
inline Matrix Matrix::sinh() const { return ocl::sinh(*this); }
inline Matrix Matrix::cosh() const { return ocl::cosh(*this); }
inline Matrix Matrix::exp() const { return ocl::exp(*this); }
inline Matrix Matrix::log() const { return ocl::log(*this); }

inline Matrix Matrix::cpow(const Matrix& exponent) const {
  return ocl::cpow(*this, exponent);
}

inline Matrix Matrix::norm() const { return ocl::norm(*this); }
inline Matrix Matrix::sum() const { return ocl::sum(*this); }
inline Matrix Matrix::min() const { return ocl::min(*this); }
inline Matrix Matrix::max() const { return ocl::max(*this); }
inline Matrix Matrix::mean() const { return ocl::mean(*this); }
inline Matrix Matrix::trace() const { return ocl::trace(*this); }

inline Matrix Matrix::reshape(const Integer rows, const Integer cols) const {
  return ocl::reshape(*this, rows, cols);
}
inline Matrix Matrix::transpose() const { return ocl::transpose(*this); }

inline Matrix Matrix::slice(const std::vector<int>& slice1, const std::vector<int>& slice2) const {
  return ocl::slice(*this, slice1, slice2);
}

inline Matrix Matrix::ctimes(const Matrix& other) const { return ocl::ctimes(*this, other); }
inline Matrix Matrix::plus(const Matrix& other) const { return ocl::plus(*this, other); }
inline Matrix Matrix::cdivide(const Matrix& other) const { return ocl::cdivide(*this, other); }
inline Matrix Matrix::minus(const Matrix& other) const { return ocl::minus(*this, other); }

inline Matrix Matrix::cmin(const Matrix& other) const { return ocl::cmin(*this, other); }
inline Matrix Matrix::cmax(const Matrix& other) const { return ocl::cmax(*this, other); }

inline Matrix Matrix::times(const Matrix& other) const { return ocl::times(*this, other); }
inline Matrix Matrix::cross(const Matrix& other) const { return ocl::cross(*this, other); }
inline Matrix Matrix::dot(const Matrix& other) const { return ocl::dot(*this, other); }

inline Matrix Matrix::atan2(const Matrix& other) const { return ocl::atan2(*this, other); }

}
#endif // OCLCPP_OCL_MATRIX_H_
