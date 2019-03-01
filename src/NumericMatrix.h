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
// Forward declaration (class defined below)
class NumericMatrix;

// Function declarations (defined in cc)
NumericMatrix uplus(const NumericMatrix& m);
NumericMatrix uminus(const NumericMatrix& m);
NumericMatrix square(const NumericMatrix& m);
NumericMatrix inverse(const NumericMatrix& m);
NumericMatrix abs(const NumericMatrix& m);
NumericMatrix sqrt(const NumericMatrix& m);
NumericMatrix sin(const NumericMatrix& m);
NumericMatrix cos(const NumericMatrix& m);
NumericMatrix tan(const NumericMatrix& m);
NumericMatrix atan(const NumericMatrix& m);
NumericMatrix asin(const NumericMatrix& m);
NumericMatrix acos(const NumericMatrix& m);
NumericMatrix tanh(const NumericMatrix& m);
NumericMatrix sinh(const NumericMatrix& m);
NumericMatrix cosh(const NumericMatrix& m);
NumericMatrix exp(const NumericMatrix& m);
NumericMatrix log(const NumericMatrix& m);

NumericMatrix pow(const NumericMatrix& m, const Scalar exponent);

NumericMatrix norm(const NumericMatrix& m) ;
NumericMatrix sum(const NumericMatrix& m);
NumericMatrix min(const NumericMatrix& m);
NumericMatrix max(const NumericMatrix& m);
NumericMatrix mean(const NumericMatrix& m);
NumericMatrix trace(const NumericMatrix& m);
NumericMatrix prod(const NumericMatrix& m);

NumericMatrix reshape(const NumericMatrix& m, const Integer rows, const Integer cols);
NumericMatrix transpose(const NumericMatrix& m);
NumericMatrix block(const NumericMatrix& m, const Integer i, const Integer j, const Integer k, const Integer l);
NumericMatrix slice(const NumericMatrix& m, const Integer i, const Integer k);

NumericMatrix ctimes(const NumericMatrix& m1, const NumericMatrix& m2);
NumericMatrix cplus(const NumericMatrix& m1, const NumericMatrix& m2);
NumericMatrix cdiv(const NumericMatrix& m1, const NumericMatrix& m2);
NumericMatrix cminus(const NumericMatrix& m1, const NumericMatrix& m2);

NumericMatrix times(const NumericMatrix& m1, const NumericMatrix& m2);
NumericMatrix cross(const NumericMatrix& m1, const NumericMatrix& m2);
NumericMatrix dot(const NumericMatrix& m1, const NumericMatrix& m2);

class NumericMatrix
{
public:

  static NumericMatrix Zero(Eigen::Index rows, Eigen::Index cols) {
    return NumericMatrix(EigenMatrixX::Zero(rows,cols));
  }

  NumericMatrix(EigenMatrixX m) : m(m) { }

  NumericMatrix uplus() { return ocl::uplus(*this); }
  NumericMatrix uminus() { return ocl::uminus(*this); }
  NumericMatrix square() { return ocl::square(*this); }
  NumericMatrix inverse() { return ocl::inverse(*this); }
  NumericMatrix abs() { return ocl::abs(*this); }
  NumericMatrix sqrt() { return ocl::sqrt(*this); }
  NumericMatrix sin() { return ocl::sin(*this); }
  NumericMatrix cos() { return ocl::cos(*this); }
  NumericMatrix tan() { return ocl::tan(*this); }
  NumericMatrix atan() { return ocl::atan(*this); }
  NumericMatrix asin() { return ocl::asin(*this); }
  NumericMatrix acos() { return ocl::acos(*this); }
  NumericMatrix tanh() { return ocl::tanh(*this); }
  NumericMatrix sinh() { return ocl::sinh(*this); }
  NumericMatrix cosh() { return ocl::cosh(*this); }
  NumericMatrix exp() { return ocl::exp(*this); }
  NumericMatrix log() { return ocl::log(*this); }

  NumericMatrix pow(const Scalar exponent) {
    return ocl::pow(*this, exponent);
  }

  NumericMatrix norm() { return ocl::norm(*this); }
  NumericMatrix sum() { return ocl::sum(*this); }
  NumericMatrix min() { return ocl::min(*this); }
  NumericMatrix max() { return ocl::max(*this); }
  NumericMatrix mean() { return ocl::mean(*this); }
  NumericMatrix trace() { return ocl::trace(*this); }
  NumericMatrix prod() { return ocl::prod(*this); }

  NumericMatrix reshape(const Integer rows, const Integer cols) {
    return ocl::reshape(*this, rows, cols);
  }
  NumericMatrix transpose() { return ocl::transpose(*this); }
  NumericMatrix block(const Integer i, const Integer j, const Integer k, const Integer l) {
    return ocl::block(*this, i, j, k, l);
  }
  NumericMatrix slice(const Integer i, const Integer k) {
    return ocl::slice(*this, i, k);
  }

  NumericMatrix ctimes(const NumericMatrix& other) { return ocl::ctimes(*this, other); }
  NumericMatrix cplus(const NumericMatrix& other) { return ocl::cplus(*this, other); }
  NumericMatrix cdiv(const NumericMatrix& other) { return ocl::cdiv(*this, other); }
  NumericMatrix cminus(const NumericMatrix& other) { return ocl::cminus(*this, other); }

  NumericMatrix times(const NumericMatrix& other) { return ocl::times(*this, other); }
  NumericMatrix cross(const NumericMatrix& other) { return ocl::cross(*this, other); }
  NumericMatrix dot(const NumericMatrix& other) { return ocl::dot(*this, other); }

  EigenMatrixX m;

}; // class NumericMatrix




} // namespace ocl
#endif // OCLCPP_OCL_EIGENMATRIX_H_
