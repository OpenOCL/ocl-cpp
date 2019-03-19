
void main()
{

  auto sys = ocl::System(&variables, &equations, NULL);

  ocl::TT x = ocl::TT(sys.states_struct, 0);
  ocl::TT z = ocl::TT(sys.algvars_struct, 0);
  ocl::TT u = ocl::TT(sys.controls_struct, 0);
  ocl::TT p = ocl::TT(sys.parameters_struct, 0);

  ocl::IE eq = sys.eval(x, z, u, p);

  ocl::TT x = ocl::SymTT(sys.states_struct);
  ocl::TT z = ocl::SymTT(sys.algvars_struct);
  ocl::TT u = ocl::SymTT(sys.controls_struct);
  ocl::TT p = ocl::SymTT(sys.parameters_struct);

  ocl::IE eq = sys.sym_eval(x, z, u, p);

}

void variables(ocl::SH& sh)
{

  sh.addState("p", Bounds(-5, 5));
  sh.addState("theta", Bounds(-2*ocl::pi, 2*ocl::pi));
  sh.addState("v");
  sh.addState("omega");

  sh.addConstrol("F", Bounds(-20, 20));

}

void equations(ocl::IEH& eh, const ocl::TT& x, const ocl::TT& z, const ocl::TT& u, const ocl::TT& p)
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
