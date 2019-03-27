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
#include <utils/testing.h>
#include "system.h"
#include "utils/constants.h"

void vars01Pendulum(ocl::SVH& sh);
void eq01Pendulum(ocl::SEH& eh, const ocl::TT& x, const ocl::TT& z, const ocl::TT& u, const ocl::TT& p);

TEST(System, aSystemEvaluation)
{
  auto sys = ocl::System(&vars01Pendulum, &eq01Pendulum);

  ocl::Matrix x = ocl::Matrix::Zero(2,1);
  ocl::Matrix z = ocl::Matrix::Zero(0,1);
  ocl::Matrix u = ocl::Matrix::Zero(1,1);
  ocl::Matrix p = ocl::Matrix::Zero(0,1);

  ocl::Matrix diff_out;
  ocl::Matrix implicit_out;
  sys.evaluate(x, z, u, p, diff_out, implicit_out);

  ocl::test::assertEqual( ocl::full(diff_out), {{0,-9.8}}, OCL_INFO);
}

void vars01Pendulum(ocl::SVH& sh)
{
  sh.state("p", {1,1}, -5, 5);
  sh.state("v");

  sh.control("F", {1,1}, -20, 20);
}

void eq01Pendulum(ocl::SEH& eh, const ocl::TT& x, const ocl::TT& z, const ocl::TT& u, const ocl::TT& p)
{
  ocl::Tensor g = 9.8;

  ocl::Tensor x_p = x.get("p").value();
  ocl::Tensor x_v = x.get("v").value();

  ocl::Tensor u_F = u.get("F").value();

  auto a = -g + u_F;

  eh.differentialEquation("p", x_v);
  eh.differentialEquation("v", a);

  // suppress warning of unused z, p
  (void) z;
  (void) p;
}
