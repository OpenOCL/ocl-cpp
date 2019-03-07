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
static inline CasadiMatrixNat norm(const CasadiMatrixNat& m) {
  return CasadiMatrixNat::norm_2(m);
}
static inline CasadiMatrixNat sum(const CasadiMatrixNat& m) { return CasadiMatrixNat::sum1(CasadiMatrixNat::sum2(m)); }
static inline CasadiMatrixNat min(const CasadiMatrixNat& m) { return CasadiMatrixNat::mmin(m); }
static inline CasadiMatrixNat max(const CasadiMatrixNat& m) { return CasadiMatrixNat::mmax(m); }
static inline CasadiMatrixNat mean(const CasadiMatrixNat& m) { return sum(m)/(m.rows()*m.columns()); }
static inline CasadiMatrixNat trace(const CasadiMatrixNat& m) { return CasadiMatrixNat::trace(m); }

// geometrical
static inline CasadiMatrixNat reshape(const CasadiMatrixNat& m, CasadiInteger rows, CasadiInteger cols) {
  return CasadiMatrixNat::reshape(m, rows, cols);
}
static inline CasadiMatrixNat transpose(const CasadiMatrixNat& m) { return m.T(); }

// get block slice of cols (i:j) and rows (k:l)
static inline CasadiMatrixNat block(const CasadiMatrixNat& m, CasadiInteger i, CasadiInteger j, CasadiInteger k, CasadiInteger l) {
  ::casadi::Slice s1(i,j);
  ::casadi::Slice s2(k,l);
  CasadiMatrixNat ret;
  m.get(ret, false, s1, s2);
  return ret;
}
// get element at (i,k)
static inline CasadiMatrixNat slice(const CasadiMatrixNat& m, CasadiInteger i, CasadiInteger k) {
  return block(m,i,i,k,k);
}

// binary coefficient wise
static inline CasadiMatrixNat ctimes(const CasadiMatrixNat& m1, const CasadiMatrixNat& m2) {
  return CasadiMatrixNat::times(m1, m2);
}
static inline CasadiMatrixNat plus(const CasadiMatrixNat& m1, const CasadiMatrixNat& m2) {
  return m1 + m2;
}
static inline CasadiMatrixNat cdiv(const CasadiMatrixNat& m1, const CasadiMatrixNat& m2) {
  return CasadiMatrixNat::mrdivide(m1,m2);
}
static inline CasadiMatrixNat minus(const CasadiMatrixNat& m1, const CasadiMatrixNat& m2) {
  return m1 - m2;
}

// binary operations
static inline CasadiMatrixNat times(const CasadiMatrixNat& m1, const CasadiMatrixNat& m2) {
  return CasadiMatrixNat::mtimes(m1,m2);
}

static inline CasadiMatrixNat cross(const CasadiMatrixNat& m1, const CasadiMatrixNat& m2) {
  return CasadiMatrixNat::cross(m1, m2);
}

static inline CasadiMatrixNat dot(const CasadiMatrixNat& m1, const CasadiMatrixNat& m2) {
  return CasadiMatrixNat::dot(m1, m2);
}

}
} // namespace ocl
#endif // OCL_CASADI_H_
