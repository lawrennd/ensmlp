#include <math.h>
#include <stdlib.h>
#include "mex.h"
#include "string.h"

#define USE_MATLAB 0
#define USE_APPROX 1
#define USE_C 2
#define ERFTYPE USE_APPROX
#define SIGN(a) ((a) > 0.0 ? 1 : ((a)==0.0 ? 0 : -1))
#define NONE 0
#define DIAG 1
#define OTHER 2
#define REAPPROX 1
#define LOWER  1.160618371190047
#define UPPER  1.273239525532378 
#define APPROX  1.238228217053698
#define CONSTC  APPROX
#define PI 3.141592653589793108624468950438
#define ITMAX 100
#define EPS 3.0e-7
#define FPMIN 1.0e-30
#define SQUARE(a) ((a)*(a))


double erfcc();
double erf();

/************************************************************
                           Code for ens
************************************************************/

typedef struct {
  int numIn;
  int numOut;
  int numHidden; 
  int numWeights;
  int numParams;
  int covarStruct;

  double *w1, *b1, *w2, *b2;
  double *d1, *db1, *d2, *db2;
} ens;

void ens_initialise(ens *, int, int, int, int); 
void ens_zeroParameters(ens *);
                    
