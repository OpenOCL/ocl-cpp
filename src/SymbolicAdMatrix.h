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

#include "casadi.h"

namespace ocl
{

// Symbolic automatic differentiable matrix
class SymbolicAdMatrix
{
  typedef casadi::SX CasadiMatrixNat; // native casadi type
  typedef casadi::SX CasadiScalar;
  typedef int CasadiInteger;

  // native casadi type operations
  static inline CasadiMatrixNat m_uplus(const CasadiMatrixNat& m) { return m; }
  static inline CasadiMatrixNat m_uminus(const CasadiMatrixNat& m) { return -m; }
  static inline CasadiMatrixNat m_square(const CasadiMatrixNat& m) { return m.sq(); }
  static inline CasadiMatrixNat m_inverse(const CasadiMatrixNat& m) { return m.inv(); }
  static inline CasadiMatrixNat m_abs(const CasadiMatrixNat& m) { return m.abs(); }
  static inline CasadiMatrixNat m_sqrt(const CasadiMatrixNat& m) { return m.sqrt(); }
  static inline CasadiMatrixNat m_sin(const CasadiMatrixNat& m) { return m.sin(); }
  static inline CasadiMatrixNat m_cos(const CasadiMatrixNat& m) { return m.cos(); }
  static inline CasadiMatrixNat m_tan(const CasadiMatrixNat& m) { return m.tan(); }
  static inline CasadiMatrixNat m_atan(const CasadiMatrixNat& m) { return m.atan(); }
  static inline CasadiMatrixNat m_asin(const CasadiMatrixNat& m) { return m.asin(); }
  static inline CasadiMatrixNat m_acos(const CasadiMatrixNat& m) { return m.acos(); }
  static inline CasadiMatrixNat m_tanh(const CasadiMatrixNat& m) { return m.tanh(); }
  static inline CasadiMatrixNat m_sinh(const CasadiMatrixNat& m) { return m.sinh(); }
  static inline CasadiMatrixNat m_cosh(const CasadiMatrixNat& m) { return m.cosh(); }
  static inline CasadiMatrixNat m_exp(const CasadiMatrixNat& m) { return m.exp(); }
  static inline CasadiMatrixNat m_log(const CasadiMatrixNat& m) { return m.log(); }

  static inline CasadiMatrixNat m_pow(const CasadiMatrixNat& m, const CasadiScalar exponent) { return m.pow(exponent); }

  // reduction
  static inline CasadiMatrixNat m_norm(const CasadiMatrixNat& m) { return m.norm2(); }
  static inline CasadiMatrixNat m_sum(const CasadiMatrixNat& m) { return m.sum(); }
  static inline CasadiMatrixNat m_min(const CasadiMatrixNat& m) { return m.fmin(); }
  static inline CasadiMatrixNat m_max(const CasadiMatrixNat& m) { return m.fmax(); }
  static inline CasadiMatrixNat m_mean(const CasadiMatrixNat& m) { return m.mean(); }
  static inline CasadiMatrixNat m_trace(const CasadiMatrixNat& m) { return m.trace(); }
  static inline CasadiMatrixNat m_prod(const CasadiMatrixNat& m) { return m.prod(); }

  // geometrical
  static inline CasadiMatrixNat m_reshape(const CasadiMatrixNat& m, CasadiInteger rows, CasadiInteger cols) {
    return m.reshape(rows,cols);
  }
  static inline CasadiMatrixNat m_transpose(const CasadiMatrixNat& m) { return m.transpose(); }

  // get block slice of cols (i:j) and rows (k:l)
  static inline CasadiMatrixNat m_block(const CasadiMatrixNat& m, CasadiInteger i, CasadiInteger j, CasadiInteger k, CasadiInteger l) {
    casadi.Slice s1(i,j);
    casado.Slice s2(k,l);
    CasadiMatrixNat ret;
    m.get(ret, false, s1, s2)
    return ret;
  }
  // get element at (i,k)
  static inline CasadiMatrixNat m_slice(const CasadiMatrixNat& m, CasadiInteger i, CasadiInteger k) {
    return m_block(m,i,i,k,k);
  }

  // binary coefficient wise
  static inline CasadiMatrixNat m_ctimes(const CasadiMatrixNat& m1, const CasadiMatrixNat& m2) {
    return CasadiMatrixNat::times(m1, m2);
  }
  static inline CasadiMatrixNat m_cplus(const CasadiMatrixNat& m1, const CasadiMatrixNat& m2) {
    return m1 + m2;
  }
  static inline CasadiMatrixNat m_cdiv(const CasadiMatrixNat& m1, const CasadiMatrixNat& m2) {
    return CasadiMatrixNat::mrdivide(m1,m2);
  }
  static inline CasadiMatrixNat m_cminus(const CasadiMatrixNat& m1, const CasadiMatrixNat& m2) {
    return m1 - m2;
  }

  // binary operations
  static inline CasadiMatrixNat m_times(const CasadiMatrixNat& m1, const CasadiMatrixNat& m2) {
    return CasadiMatrixNat::mtimes(m1,m2);
  }

  static inline CasadiMatrixNat m_cross(const CasadiMatrixNat& m1, const CasadiMatrixNat& m2) {
    return CasadiMatrixNat::cross(m1, m2);
  }

  static inline CasadiMatrixNat m_dot(const CasadiMatrixNat& m1, const CasadiMatrixNat& m2) {
    return CasadiMatrixNat::dot(m1, m2);
  }

};

}
#endif // OCLCPP_OCL_SYMBOLIC_AD_MATRIX_H_
