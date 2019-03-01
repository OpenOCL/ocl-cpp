#include "mex.h"
#include "matrix.h"


static std::vector<mxArray*> instances = {};
static int numInstances = 0;

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{

  // convert inputs to c type

  // first input is always function name as string
  mxArray* fcn_name_mx = prhs[0];
  size_t n_chars = mxGetN(fcn_name_mx);
  char fcn_name[n_chars+1];;
  mxGetString(fcn_name_mx, fcn_name)

  // 
  if (strcmp(fcn_name,'create') == 0) 
  {

  } 
  else if (strcmp(fcn_name,'uplus') == 0) 
  {

  } 
  else if (strcmp(fcn_name,'plus') == 0)
  {

  }
  else if (strcmp(fcn_name,'pow') == 0)
  {

  }
  else if (strcmp(fcn_name,'reshape') == 0)
  {

  }
  else if (strcmp(fcn_name,'str') == 0)
  {

  }


}
