#include "mex.h"
#include "matrix.h"


static std::vector<mxArray*> instances = {};
static int numInstances = 0;

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{

  // convert inputs to c type

  // first input is function name as string
  mxArray* fcn_name_mx = prhs[0];
  size_t n_chars = mxGetN(fcn_name_mx);
  char fcn_name[n_chars+1];;
  mxGetString(fcn_name_mx, fcn_name)

  //


}
