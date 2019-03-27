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

  ocl::Matrix diff_out;
  ocl::Matrix implicit_out;
  sys.evaluate(0, 0, 0, 0, diff_out, implicit_out);
  //
  // ocl::test::assertEqual( ocl::full(diff_out), {{0,0,0,0}}, OCL_INFO);
}

void vars01Pendulum(ocl::SVH& sh)
{
  sh.state("p", {1,1}, -5, 5);
  sh.state("theta", {1,1}, -2*ocl::pi, 2*ocl::pi);
  sh.state("v");
  sh.state("omega");

  sh.control("F", {1,1}, -20, 20);
}

void eq01Pendulum(ocl::SEH& eh, const ocl::TT& x, const ocl::TT& z, const ocl::TT& u, const ocl::TT& p)
{
  ocl::Tensor g = 9.8;
  ocl::Tensor cm = 1.0;  // cart mass
  ocl::Tensor pm = 0.1;  // pole mass
  ocl::Tensor phl = 0.5; // pole half length

  ocl::Tensor m = cm + pm;
  ocl::Tensor pml = pm * phl; // pole mass length

  ocl::Tensor x_p = x.get("p").value();
  ocl::Tensor x_theta = x.get("theta").value();
  ocl::Tensor x_v = x.get("v").value();
  ocl::Tensor x_omega = x.get("omega").value();

  ocl::Tensor u_F = u.get("F").value();


  auto ctheta = ocl::cos( x_theta );
  auto stheta = ocl::sin( x_theta );

  auto domega = (stheta*g + ctheta * (u_F - pml * x_omega.square() * stheta) / m) /
                (phl * (ocl::Tensor(4.0/3.0) - pm * ctheta.square() / m ));

  auto a = (u_F + pml* (x_omega.square() * stheta - domega * ctheta)) / m;

  eh.differentialEquation("p", x_v);
  eh.differentialEquation("theta", x_omega);
  eh.differentialEquation("v", a);
  eh.differentialEquation("omega", domega);

  // suppress warning of unused z, p
  (void) z;
  (void) p;
}
