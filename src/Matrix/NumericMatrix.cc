#include "NumericMatrix.h"

namespace ocl
{

NumericMatrix uplus(const NumericMatrix& m) { return NumericMatrix(eigen::uplus(m.m)); }
NumericMatrix uminus(const NumericMatrix& m) { return NumericMatrix(eigen::uminus(m.m)); }
NumericMatrix square(const NumericMatrix& m) { return NumericMatrix(eigen::square(m.m)); }
NumericMatrix inverse(const NumericMatrix& m) { return NumericMatrix(eigen::inverse(m.m)); }
NumericMatrix abs(const NumericMatrix& m) { return NumericMatrix(eigen::abs(m.m)); }
NumericMatrix sqrt(const NumericMatrix& m) { return NumericMatrix(eigen::sqrt(m.m)); }
NumericMatrix sin(const NumericMatrix& m) { return NumericMatrix(eigen::sin(m.m)); }
NumericMatrix cos(const NumericMatrix& m) { return NumericMatrix(eigen::cos(m.m)); }
NumericMatrix tan(const NumericMatrix& m) { return NumericMatrix(eigen::tan(m.m)); }
NumericMatrix atan(const NumericMatrix& m) { return NumericMatrix(eigen::atan(m.m)); }
NumericMatrix asin(const NumericMatrix& m) { return NumericMatrix(eigen::asin(m.m)); }
NumericMatrix acos(const NumericMatrix& m) { return NumericMatrix(eigen::acos(m.m)); }
NumericMatrix tanh(const NumericMatrix& m) { return NumericMatrix(eigen::tanh(m.m)); }
NumericMatrix sinh(const NumericMatrix& m) { return NumericMatrix(eigen::sinh(m.m)); }
NumericMatrix cosh(const NumericMatrix& m) { return NumericMatrix(eigen::cosh(m.m)); }
NumericMatrix exp(const NumericMatrix& m) { return NumericMatrix(eigen::exp(m.m)); }
NumericMatrix log(const NumericMatrix& m) { return NumericMatrix(eigen::log(m.m)); }

NumericMatrix pow(const NumericMatrix& m, const Scalar exponent) { return NumericMatrix(eigen::pow(m.m, exponent)); }

NumericMatrix norm(const NumericMatrix& m) { return NumericMatrix(eigen::norm(m.m)); }
NumericMatrix sum(const NumericMatrix& m) { return NumericMatrix(eigen::sum(m.m)); }
NumericMatrix min(const NumericMatrix& m) { return NumericMatrix(eigen::min(m.m)); }
NumericMatrix max(const NumericMatrix& m) { return NumericMatrix(eigen::max(m.m)); }
NumericMatrix mean(const NumericMatrix& m) { return NumericMatrix(eigen::mean(m.m)); }
NumericMatrix trace(const NumericMatrix& m) { return NumericMatrix(eigen::trace(m.m)); }
NumericMatrix prod(const NumericMatrix& m) { return NumericMatrix(eigen::prod(m.m)); }

NumericMatrix reshape(const NumericMatrix& m, const Integer rows, const Integer cols) {
  return NumericMatrix(eigen::reshape(m.m, rows, cols));
}
NumericMatrix transpose(const NumericMatrix& m) {
  return NumericMatrix(eigen::transpose(m.m));
}
NumericMatrix block(const NumericMatrix& m, const Integer i, const Integer j, const Integer k, const Integer l) {
  return NumericMatrix(eigen::block(m.m, i, j, k, l));
}
NumericMatrix slice(const NumericMatrix& m, const Integer i, const Integer k) {
  return NumericMatrix(eigen::slice(m.m, i, k));
}

NumericMatrix ctimes(const NumericMatrix& m1, const NumericMatrix& m2) { return NumericMatrix(eigen::ctimes(m1.m, m2.m)); }
NumericMatrix cplus(const NumericMatrix& m1, const NumericMatrix& m2) { return NumericMatrix(eigen::cplus(m1.m, m2.m)); }
NumericMatrix cdiv(const NumericMatrix& m1, const NumericMatrix& m2) { return NumericMatrix(eigen::cdiv(m1.m, m2.m)); }
NumericMatrix cminus(const NumericMatrix& m1, const NumericMatrix& m2) { return NumericMatrix(eigen::cminus(m1.m, m2.m)); }

NumericMatrix times(const NumericMatrix& m1, const NumericMatrix& m2) { return NumericMatrix(eigen::times(m1.m, m2.m)); }
NumericMatrix cross(const NumericMatrix& m1, const NumericMatrix& m2) { return NumericMatrix(eigen::cross(m1.m, m2.m)); }
NumericMatrix dot(const NumericMatrix& m1, const NumericMatrix& m2) { return NumericMatrix(eigen::dot(m1.m, m2.m)); }

}
