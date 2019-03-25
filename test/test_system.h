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

TEST(System, aVariablesHandler)
{
  ocl::SystemVariablesHandler sh;

  sh.state("p", Bounds(-5, 5));
  sh.state("theta", Bounds(-2*ocl::pi, 2*ocl::pi));
  sh.state("v");
  sh.state("omega");

  sh.control("F", Bounds(-20, 20));
}

TEST(System, bSystemConstruction)
{
  auto sys = ocl::System(&vars01Pendulum, &eq01Pendulum);
}

TEST(System, cSystemEvaluation)
{
  auto sys = ocl::System(&vars01Pendulum, &eq01Pendulum);
  ocl::IE eq = sys.eval(0, 0, 0, 0);

  ocl::TT x = ocl::SymTT(sys.states_struct);
  ocl::TT z = ocl::SymTT(sys.algvars_struct);
  ocl::TT u = ocl::SymTT(sys.controls_struct);
  ocl::TT p = ocl::SymTT(sys.parameters_struct);

  ocl::IE eq = sys.sym_eval(x, z, u, p);
}

void vars01Pendulum(ocl::SVH& sh)
{

  sh.addState("p", Bounds(-5, 5));
  sh.addState("theta", Bounds(-2*ocl::pi, 2*ocl::pi));
  sh.addState("v");
  sh.addState("omega");

  sh.addConstrol("F", Bounds(-20, 20));

}

void eq01Pendulum(ocl::SEH& eh, const ocl::TT& x, const ocl::TT& z, const ocl::TT& u, const ocl::TT& p)
{

  double g = 9.8;
  double cm = 1.0;  // cart mass
  double pm = 0.1;  // pole mass
  double phl = 0.5; // pole half length

  double m = cm + pm;
  double pml = pm * phl; // pole mass length

  auto ctheta = ocl::cos( x("theta") );
  auto stheta = ocl::sin( x("theta") );

  auto domega = (g*stheta + ctheta * (-u("F") - pml * x("omega").square() * stheta) / m) /
                (phl * (4.0/3.0 - pm * ctheta.square() / m));

  auto a = (u.F + pml* (x("omega").square() * stheta - domega * ctheta)) / m;

  eh.setODE("p", x("v"));
  eh.setODE("theta", x("omega"));
  eh.setODE("v", a);
  eh.setODE("omega", domega);

}
