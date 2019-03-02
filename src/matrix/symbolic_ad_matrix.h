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

namespace ocl
{

// Symbolic automatic differentiable matrix
class SymbolicAdMatrix
{
public:

  SymbolicAdMatrix(CasadiMatrixNat m) : m(m) { }

  // Member functions are defined inline below class (after static functions).
  SymbolicAdMatrix uplus();
  SymbolicAdMatrix uminus();
  SymbolicAdMatrix square();
  SymbolicAdMatrix inverse();
  SymbolicAdMatrix abs();
  SymbolicAdMatrix sqrt();
  SymbolicAdMatrix sin();
  SymbolicAdMatrix cos();
  SymbolicAdMatrix tan();
  SymbolicAdMatrix atan();
  SymbolicAdMatrix asin();
  SymbolicAdMatrix acos();
  SymbolicAdMatrix tanh();
  SymbolicAdMatrix sinh();
  SymbolicAdMatrix cosh();
  SymbolicAdMatrix exp();
  SymbolicAdMatrix log();

  SymbolicAdMatrix pow(const Scalar exponent);

  SymbolicAdMatrix norm();
  SymbolicAdMatrix sum();
  SymbolicAdMatrix min();
  SymbolicAdMatrix max();
  SymbolicAdMatrix mean();
  SymbolicAdMatrix trace();
  SymbolicAdMatrix prod();

  SymbolicAdMatrix reshape(const Integer rows, const Integer cols);
  SymbolicAdMatrix transpose();
  SymbolicAdMatrix block(const Integer i, const Integer j, const Integer k, const Integer l);
  SymbolicAdMatrix slice(const Integer i, const Integer k);

  SymbolicAdMatrix ctimes(const SymbolicAdMatrix& other);
  SymbolicAdMatrix cplus(const SymbolicAdMatrix& other);
  SymbolicAdMatrix cdiv(const SymbolicAdMatrix& other);
  SymbolicAdMatrix cminus(const SymbolicAdMatrix& other);

  SymbolicAdMatrix times(const SymbolicAdMatrix& other);
  SymbolicAdMatrix cross(const SymbolicAdMatrix& other);
  SymbolicAdMatrix dot(const SymbolicAdMatrix& other);

  CasadiMatrixNat m;
};

// Static functions
static inline SymbolicAdMatrix uplus(const SymbolicAdMatrix& m) { return SymbolicAdMatrix(casadi::uplus(m.m)); }
static inline SymbolicAdMatrix uminus(const SymbolicAdMatrix& m) { return SymbolicAdMatrix(casadi::uminus(m.m)); }
static inline SymbolicAdMatrix square(const SymbolicAdMatrix& m) { return SymbolicAdMatrix(casadi::square(m.m)); }
static inline SymbolicAdMatrix inverse(const SymbolicAdMatrix& m) { return SymbolicAdMatrix(casadi::inverse(m.m)); }
static inline SymbolicAdMatrix abs(const SymbolicAdMatrix& m) { return SymbolicAdMatrix(casadi::abs(m.m)); }
static inline SymbolicAdMatrix sqrt(const SymbolicAdMatrix& m) { return SymbolicAdMatrix(casadi::sqrt(m.m)); }
static inline SymbolicAdMatrix sin(const SymbolicAdMatrix& m) { return SymbolicAdMatrix(casadi::sin(m.m)); }
static inline SymbolicAdMatrix cos(const SymbolicAdMatrix& m) { return SymbolicAdMatrix(casadi::cos(m.m)); }
static inline SymbolicAdMatrix tan(const SymbolicAdMatrix& m) { return SymbolicAdMatrix(casadi::tan(m.m)); }
static inline SymbolicAdMatrix atan(const SymbolicAdMatrix& m) { return SymbolicAdMatrix(casadi::atan(m.m)); }
static inline SymbolicAdMatrix asin(const SymbolicAdMatrix& m) { return SymbolicAdMatrix(casadi::asin(m.m)); }
static inline SymbolicAdMatrix acos(const SymbolicAdMatrix& m) { return SymbolicAdMatrix(casadi::acos(m.m)); }
static inline SymbolicAdMatrix tanh(const SymbolicAdMatrix& m) { return SymbolicAdMatrix(casadi::tanh(m.m)); }
static inline SymbolicAdMatrix sinh(const SymbolicAdMatrix& m) { return SymbolicAdMatrix(casadi::sinh(m.m)); }
static inline SymbolicAdMatrix cosh(const SymbolicAdMatrix& m) { return SymbolicAdMatrix(casadi::cosh(m.m)); }
static inline SymbolicAdMatrix exp(const SymbolicAdMatrix& m) { return SymbolicAdMatrix(casadi::exp(m.m)); }
static inline SymbolicAdMatrix log(const SymbolicAdMatrix& m) { return SymbolicAdMatrix(casadi::log(m.m)); }

static inline SymbolicAdMatrix pow(const SymbolicAdMatrix& m, const Scalar exponent) { return SymbolicAdMatrix(casadi::pow(m.m, exponent)); }

static inline SymbolicAdMatrix norm(const SymbolicAdMatrix& m) { return SymbolicAdMatrix(casadi::norm(m.m)); }
static inline SymbolicAdMatrix sum(const SymbolicAdMatrix& m) { return SymbolicAdMatrix(casadi::sum(m.m)); }
static inline SymbolicAdMatrix min(const SymbolicAdMatrix& m) { return SymbolicAdMatrix(casadi::min(m.m)); }
static inline SymbolicAdMatrix max(const SymbolicAdMatrix& m) { return SymbolicAdMatrix(casadi::max(m.m)); }
static inline SymbolicAdMatrix mean(const SymbolicAdMatrix& m) { return SymbolicAdMatrix(casadi::mean(m.m)); }
static inline SymbolicAdMatrix trace(const SymbolicAdMatrix& m) { return SymbolicAdMatrix(casadi::trace(m.m)); }
static inline SymbolicAdMatrix prod(const SymbolicAdMatrix& m) { return SymbolicAdMatrix(casadi::prod(m.m)); }

static inline SymbolicAdMatrix reshape(const SymbolicAdMatrix& m, const Integer rows, const Integer cols) {
  return SymbolicAdMatrix(casadi::reshape(m.m, rows, cols));
}
static inline SymbolicAdMatrix transpose(const SymbolicAdMatrix& m) {
  return SymbolicAdMatrix(casadi::transpose(m.m));
}
static inline SymbolicAdMatrix block(const SymbolicAdMatrix& m, const Integer i, const Integer j, const Integer k, const Integer l) {
  return SymbolicAdMatrix(casadi::block(m.m, i, j, k, l));
}
static inline SymbolicAdMatrix slice(const SymbolicAdMatrix& m, const Integer i, const Integer k) {
  return SymbolicAdMatrix(casadi::slice(m.m, i, k));
}

static inline SymbolicAdMatrix ctimes(const SymbolicAdMatrix& m1, const SymbolicAdMatrix& m2) { return SymbolicAdMatrix(casadi::ctimes(m1.m, m2.m)); }
static inline SymbolicAdMatrix cplus(const SymbolicAdMatrix& m1, const SymbolicAdMatrix& m2) { return SymbolicAdMatrix(casadi::cplus(m1.m, m2.m)); }
static inline SymbolicAdMatrix cdiv(const SymbolicAdMatrix& m1, const SymbolicAdMatrix& m2) { return SymbolicAdMatrix(casadi::cdiv(m1.m, m2.m)); }
static inline SymbolicAdMatrix cminus(const SymbolicAdMatrix& m1, const SymbolicAdMatrix& m2) { return SymbolicAdMatrix(casadi::cminus(m1.m, m2.m)); }

static inline SymbolicAdMatrix times(const SymbolicAdMatrix& m1, const SymbolicAdMatrix& m2) { return SymbolicAdMatrix(casadi::times(m1.m, m2.m)); }
static inline SymbolicAdMatrix cross(const SymbolicAdMatrix& m1, const SymbolicAdMatrix& m2) { return SymbolicAdMatrix(casadi::cross(m1.m, m2.m)); }
static inline SymbolicAdMatrix dot(const SymbolicAdMatrix& m1, const SymbolicAdMatrix& m2) { return SymbolicAdMatrix(casadi::dot(m1.m, m2.m)); }

// Member functions (calling the static functions above)
inline SymbolicAdMatrix SymbolicAdMatrix::uplus() { return ocl::uplus(*this); }
inline SymbolicAdMatrix SymbolicAdMatrix::uminus() { return ocl::uminus(*this); }
inline SymbolicAdMatrix SymbolicAdMatrix::square() { return ocl::square(*this); }
inline SymbolicAdMatrix SymbolicAdMatrix::inverse() { return ocl::inverse(*this); }
inline SymbolicAdMatrix SymbolicAdMatrix::abs() { return ocl::abs(*this); }
inline SymbolicAdMatrix SymbolicAdMatrix::sqrt() { return ocl::sqrt(*this); }
inline SymbolicAdMatrix SymbolicAdMatrix::sin() { return ocl::sin(*this); }
inline SymbolicAdMatrix SymbolicAdMatrix::cos() { return ocl::cos(*this); }
inline SymbolicAdMatrix SymbolicAdMatrix::tan() { return ocl::tan(*this); }
inline SymbolicAdMatrix SymbolicAdMatrix::atan() { return ocl::atan(*this); }
inline SymbolicAdMatrix SymbolicAdMatrix::asin() { return ocl::asin(*this); }
inline SymbolicAdMatrix SymbolicAdMatrix::acos() { return ocl::acos(*this); }
inline SymbolicAdMatrix SymbolicAdMatrix::tanh() { return ocl::tanh(*this); }
inline SymbolicAdMatrix SymbolicAdMatrix::sinh() { return ocl::sinh(*this); }
inline SymbolicAdMatrix SymbolicAdMatrix::cosh() { return ocl::cosh(*this); }
inline SymbolicAdMatrix SymbolicAdMatrix::exp() { return ocl::exp(*this); }
inline SymbolicAdMatrix SymbolicAdMatrix::log() { return ocl::log(*this); }

inline SymbolicAdMatrix SymbolicAdMatrix::pow(const Scalar exponent) {
  return ocl::pow(*this, exponent);
}

inline SymbolicAdMatrix SymbolicAdMatrix::norm() { return ocl::norm(*this); }
inline SymbolicAdMatrix SymbolicAdMatrix::sum() { return ocl::sum(*this); }
inline SymbolicAdMatrix SymbolicAdMatrix::min() { return ocl::min(*this); }
inline SymbolicAdMatrix SymbolicAdMatrix::max() { return ocl::max(*this); }
inline SymbolicAdMatrix SymbolicAdMatrix::mean() { return ocl::mean(*this); }
inline SymbolicAdMatrix SymbolicAdMatrix::trace() { return ocl::trace(*this); }
inline SymbolicAdMatrix SymbolicAdMatrix::prod() { return ocl::prod(*this); }

inline SymbolicAdMatrix SymbolicAdMatrix::reshape(const Integer rows, const Integer cols) {
  return ocl::reshape(*this, rows, cols);
}
inline SymbolicAdMatrix SymbolicAdMatrix::transpose() { return ocl::transpose(*this); }
inline SymbolicAdMatrix SymbolicAdMatrix::block(const Integer i, const Integer j, const Integer k, const Integer l) {
  return ocl::block(*this, i, j, k, l);
}
inline SymbolicAdMatrix SymbolicAdMatrix::slice(const Integer i, const Integer k) {
  return ocl::slice(*this, i, k);
}

inline SymbolicAdMatrix SymbolicAdMatrix::ctimes(const SymbolicAdMatrix& other) { return ocl::ctimes(*this, other); }
inline SymbolicAdMatrix SymbolicAdMatrix::cplus(const SymbolicAdMatrix& other) { return ocl::cplus(*this, other); }
inline SymbolicAdMatrix SymbolicAdMatrix::cdiv(const SymbolicAdMatrix& other) { return ocl::cdiv(*this, other); }
inline SymbolicAdMatrix SymbolicAdMatrix::cminus(const SymbolicAdMatrix& other) { return ocl::cminus(*this, other); }

inline SymbolicAdMatrix SymbolicAdMatrix::times(const SymbolicAdMatrix& other) { return ocl::times(*this, other); }
inline SymbolicAdMatrix SymbolicAdMatrix::cross(const SymbolicAdMatrix& other) { return ocl::cross(*this, other); }
inline SymbolicAdMatrix SymbolicAdMatrix::dot(const SymbolicAdMatrix& other) { return ocl::dot(*this, other); }

}
#endif // OCLCPP_OCL_SYMBOLIC_AD_MATRIX_H_
