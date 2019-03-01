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

static inline NumericMatrix uplus(const NumericMatrix& m) { return NumericMatrix(ocl::eigen::uplus(m.m)); }
static inline NumericMatrix uminus(const NumericMatrix& m) { return NumericMatrix(ocl::eigen::uminus(m.m)); }
static inline NumericMatrix square(const NumericMatrix& m) { return NumericMatrix(ocl::eigen::square(m.m)); }
static inline NumericMatrix inverse(const NumericMatrix& m) { return NumericMatrix(ocl::eigen::inverse(m.m)); }
static inline NumericMatrix abs(const NumericMatrix& m) { return NumericMatrix(ocl::eigen::abs(m.m)); }
static inline NumericMatrix sqrt(const NumericMatrix& m) { return NumericMatrix(ocl::eigen::sqrt(m.m)); }
static inline NumericMatrix sin(const NumericMatrix& m) { return NumericMatrix(ocl::eigen::sin(m.m)); }
static inline NumericMatrix cos(const NumericMatrix& m) { return NumericMatrix(ocl::eigen::cos(m.m)); }
static inline NumericMatrix tan(const NumericMatrix& m) { return NumericMatrix(ocl::eigen::tan(m.m)); }
static inline NumericMatrix atan(const NumericMatrix& m) { return NumericMatrix(ocl::eigen::atan(m.m)); }
static inline NumericMatrix asin(const NumericMatrix& m) { return NumericMatrix(ocl::eigen::asin(m.m)); }
static inline NumericMatrix acos(const NumericMatrix& m) { return NumericMatrix(ocl::eigen::acos(m.m)); }
static inline NumericMatrix tanh(const NumericMatrix& m) { return NumericMatrix(ocl::eigen::tanh(m.m)); }
static inline NumericMatrix sinh(const NumericMatrix& m) { return NumericMatrix(ocl::eigen::sinh(m.m)); }
static inline NumericMatrix cosh(const NumericMatrix& m) { return NumericMatrix(ocl::eigen::cosh(m.m)); }
static inline NumericMatrix exp(const NumericMatrix& m) { return NumericMatrix(ocl::eigen::exp(m.m)); }
static inline NumericMatrix log(const NumericMatrix& m) { return NumericMatrix(ocl::eigen::log(m.m)); }

static inline NumericMatrix pow(const NumericMatrix& m, const Scalar exponent) { return NumericMatrix(ocl::eigen::pow(m.m, exponent)); }

static inline NumericMatrix norm(const NumericMatrix& m) { return NumericMatrix(ocl::eigen::norm(m.m)); }
static inline NumericMatrix sum(const NumericMatrix& m) { return NumericMatrix(ocl::eigen::sum(m.m)); }
static inline NumericMatrix min(const NumericMatrix& m) { return NumericMatrix(ocl::eigen::min(m.m)); }
static inline NumericMatrix max(const NumericMatrix& m) { return NumericMatrix(ocl::eigen::max(m.m)); }
static inline NumericMatrix mean(const NumericMatrix& m) { return NumericMatrix(ocl::eigen::mean(m.m)); }
static inline NumericMatrix trace(const NumericMatrix& m) { return NumericMatrix(ocl::eigen::trace(m.m)); }
static inline NumericMatrix prod(const NumericMatrix& m) { return NumericMatrix(ocl::eigen::prod(m.m)); }

static inline NumericMatrix reshape(const NumericMatrix& m, const Integer rows, const Integer cols) {
  return NumericMatrix(ocl::eigen::reshape(m.m, rows, cols));
}
static inline NumericMatrix transpose(const NumericMatrix& m) {
  return NumericMatrix(ocl::eigen::transpose(m.m));
}
static inline NumericMatrix block(const NumericMatrix& m, const Integer i, const Integer j, const Integer k, const Integer l) {
  return NumericMatrix(ocl::eigen::block(m.m, i, j, k, l));
}
static inline NumericMatrix slice(const NumericMatrix& m, const Integer i, const Integer k) {
  return NumericMatrix(ocl::eigen::slice(m.m, i, k);
}

static inline NumericMatrix ctimes(const NumericMatrix& m1, const NumericMatrix& m2) { return NumericMatrix(ocl::eigen::ctimes(m1.m, m2.m)); }
static inline NumericMatrix cplus(const NumericMatrix& m1, const NumericMatrix& m2) { return NumericMatrix(ocl::eigen::cplus(m1.m, m2.m)); }
static inline NumericMatrix cdiv(const NumericMatrix& m1, const NumericMatrix& m2) { return NumericMatrix(ocl::eigen::cdiv(m1.m, m2.m)); }
static inline NumericMatrix cminus(const NumericMatrix& m1, const NumericMatrix& m2) { NumericMatrix(ocl::eigen::cminus(m1.m, m2.m)); }

static inline NumericMatrix times(const NumericMatrix& m1, const NumericMatrix& m2) { NumericMatrix(ocl::eigen::times(m1.m, m2.m)); }
static inline NumericMatrix cross(const NumericMatrix& m1, const NumericMatrix& m2) { NumericMatrix(ocl::eigen::cros(m1.m, m2.m)); }
static inline NumericMatrix dot(const NumericMatrix& m1, const NumericMatrix& m2) { NumericMatrix(ocl::eigen::dot(m1.m, m2.m)); }


class NumericMatrix
{
public:
  NumericMatrix(ocl::eigen::EigenMatrixX m) : m(m) { }

  NumericMatrix uplus() { return ocl::uplus(*this)); }
  NumericMatrix uminus() { return ocl::uminus(*this)); }
  NumericMatrix square() { return ocl::square(*this)); }
  NumericMatrix inverse() { return ocl::inverse(*this)); }
  NumericMatrix abs() { return ocl::abs(*this)); }
  NumericMatrix sqrt() { return ocl::sqrt(*this)); }
  NumericMatrix sin() { return ocl::sin(*this)); }
  NumericMatrix cos() { return ocl::cos(*this)); }
  NumericMatrix tan() { return ocl::tan(*this)); }
  NumericMatrix atan() { return ocl::atan(*this)); }
  NumericMatrix asin() { return ocl::asin(*this)); }
  NumericMatrix acos() { return ocl::acos(*this)); }
  NumericMatrix tanh() { return ocl::tanh(*this)); }
  NumericMatrix sinh() { return ocl::sinh(*this)); }
  NumericMatrix cosh() { return ocl::cosh(*this)); }
  NumericMatrix exp() { return ocl::exp(*this)); }
  NumericMatrix log() { return ocl::log(*this)); }

  NumericMatrix pow(const Scalar exponent) {
    return ocl::pow(*this, exponent));
  }

  NumericMatrix norm() { return ocl::norm(*this)); }
  NumericMatrix sum() { return ocl::sum(*this)); }
  NumericMatrix min() { return ocl::min(*this)); }
  NumericMatrix max() { return ocl::max(*this)); }
  NumericMatrix mean() { return ocl::mean(*this)); }
  NumericMatrix trace() { return ocl::trace(*this)); }
  NumericMatrix prod() { return ocl::prod(*this)); }

  NumericMatrix reshape(const Integer rows, const Integer cols) {
    return return ocl::reshape(*this, rows, cols));
  }
  NumericMatrix transpose() {
    return ocl::transpose(*this));
  }
  NumericMatrix block(const Integer i, const Integer j, const Integer k, const Integer l) {
    return ocl:::block(*this, i, j, k, l);
  }
  NumericMatrix slice(const Integer i, const Integer k) {
    return ocl:::slice(*this, i, k);
  }

  NumericMatrix ctimes(const NumericMatrix& other) { return ocl::ctimes(*this, other)); }
  NumericMatrix cplus(const NumericMatrix& other) { return ocl::cplus(*this, other)); }
  NumericMatrix cdiv(const NumericMatrix& other) { return ocl::cdiv(*this, other)); }
  NumericMatrix cminus(const NumericMatrix& other) { return ocl::cminus(*this, other)); }

  NumericMatrix times(const NumericMatrix& other) { return ocl::times(*this, other)); }
  NumericMatrix cross(const NumericMatrix& other) { return ocl::cross(*this, other)); }
  NumericMatrix dot(const NumericMatrix& other) { return ocl::dot(*this, other)); }


private:
  ocl::eigen::EigenMatrixX m;

}; // class NumericMatrix




} // namespace ocl
#endif // OCLCPP_OCL_EIGENMATRIX_H_
