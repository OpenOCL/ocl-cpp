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
#ifndef OCL_CASADI_H_
#define OCL_CASADI_H_

namespace ocl
{
typedef casadi::SX CasadiMatrixNat; // native casadi type
typedef casadi::SX CasadiScalar;
typedef int CasadiInteger;

namespace casadi
{
// native casadi type operations
static inline CasadiMatrixNat uplus(const CasadiMatrixNat& m) { return m; }
static inline CasadiMatrixNat uminus(const CasadiMatrixNat& m) { return -m; }
static inline CasadiMatrixNat square(const CasadiMatrixNat& m) { return CasadiMatrixNat::sq(m); }
static inline CasadiMatrixNat inverse(const CasadiMatrixNat& m) { return CasadiMatrixNat::inv(m); }
static inline CasadiMatrixNat abs(const CasadiMatrixNat& m) { return CasadiMatrixNat::abs(m); }
static inline CasadiMatrixNat sqrt(const CasadiMatrixNat& m) { return CasadiMatrixNat::sqrt(m); }
static inline CasadiMatrixNat sin(const CasadiMatrixNat& m) { return CasadiMatrixNat::sin(m); }
static inline CasadiMatrixNat cos(const CasadiMatrixNat& m) { return CasadiMatrixNat::cos(m); }
static inline CasadiMatrixNat tan(const CasadiMatrixNat& m) { return CasadiMatrixNat::tan(m); }
static inline CasadiMatrixNat atan(const CasadiMatrixNat& m) { return CasadiMatrixNat::atan(m); }
static inline CasadiMatrixNat asin(const CasadiMatrixNat& m) { return CasadiMatrixNat::asin(m); }
static inline CasadiMatrixNat acos(const CasadiMatrixNat& m) { return CasadiMatrixNat::acos(m); }
static inline CasadiMatrixNat tanh(const CasadiMatrixNat& m) { return CasadiMatrixNat::tanh(m); }
static inline CasadiMatrixNat sinh(const CasadiMatrixNat& m) { return CasadiMatrixNat::sinh(m); }
static inline CasadiMatrixNat cosh(const CasadiMatrixNat& m) { return CasadiMatrixNat::cosh(m); }
static inline CasadiMatrixNat exp(const CasadiMatrixNat& m) { return CasadiMatrixNat::exp(m); }
static inline CasadiMatrixNat log(const CasadiMatrixNat& m) { return CasadiMatrixNat::log(m); }

static inline CasadiMatrixNat pow(const CasadiMatrixNat& m, const CasadiScalar exponent) {
  return CasadiMatrixNat::pow(m, exponent);
}



// reduction
// static inline CasadiMatrixNat m_norm(const CasadiMatrixNat& m) { return casadi::norm2(m); }
// static inline CasadiMatrixNat m_sum(const CasadiMatrixNat& m) { return m.sum(); }
// static inline CasadiMatrixNat m_min(const CasadiMatrixNat& m) { return m.fmin(); }
// static inline CasadiMatrixNat m_max(const CasadiMatrixNat& m) { return m.fmax(); }
// static inline CasadiMatrixNat m_mean(const CasadiMatrixNat& m) { return m.mean(); }
// static inline CasadiMatrixNat m_trace(const CasadiMatrixNat& m) { return m.trace(); }
// static inline CasadiMatrixNat m_prod(const CasadiMatrixNat& m) { return m.prod(); }
//
// // geometrical
// static inline CasadiMatrixNat m_reshape(const CasadiMatrixNat& m, CasadiInteger rows, CasadiInteger cols) {
//   return m.reshape(rows,cols);
// }
// static inline CasadiMatrixNat m_transpose(const CasadiMatrixNat& m) { return m.transpose(); }
//
// // get block slice of cols (i:j) and rows (k:l)
// static inline CasadiMatrixNat m_block(const CasadiMatrixNat& m, CasadiInteger i, CasadiInteger j, CasadiInteger k, CasadiInteger l) {
//   casadi.Slice s1(i,j);
//   casado.Slice s2(k,l);
//   CasadiMatrixNat ret;
//   m.get(ret, false, s1, s2)
//   return ret;
// }
// // get element at (i,k)
// static inline CasadiMatrixNat m_slice(const CasadiMatrixNat& m, CasadiInteger i, CasadiInteger k) {
//   return m_block(m,i,i,k,k);
// }
//
// // binary coefficient wise
// static inline CasadiMatrixNat m_ctimes(const CasadiMatrixNat& m1, const CasadiMatrixNat& m2) {
//   return CasadiMatrixNat::times(m1, m2);
// }
// static inline CasadiMatrixNat m_cplus(const CasadiMatrixNat& m1, const CasadiMatrixNat& m2) {
//   return m1 + m2;
// }
// static inline CasadiMatrixNat m_cdiv(const CasadiMatrixNat& m1, const CasadiMatrixNat& m2) {
//   return CasadiMatrixNat::mrdivide(m1,m2);
// }
// static inline CasadiMatrixNat m_cminus(const CasadiMatrixNat& m1, const CasadiMatrixNat& m2) {
//   return m1 - m2;
// }
//
// // binary operations
// static inline CasadiMatrixNat m_times(const CasadiMatrixNat& m1, const CasadiMatrixNat& m2) {
//   return CasadiMatrixNat::mtimes(m1,m2);
// }
//
// static inline CasadiMatrixNat m_cross(const CasadiMatrixNat& m1, const CasadiMatrixNat& m2) {
//   return CasadiMatrixNat::cross(m1, m2);
// }
//
// static inline CasadiMatrixNat m_dot(const CasadiMatrixNat& m1, const CasadiMatrixNat& m2) {
//   return CasadiMatrixNat::dot(m1, m2);
// }

}
} // namespace ocl
#endif // OCL_CASADI_H_
