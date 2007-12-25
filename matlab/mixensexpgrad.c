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

void mexFunction(
                 int nlhs,       mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]
		 )

{/*function [sg, qg] = ensexpgrad(net, x)*/
  
  mxArray *w1, *b1, *d1, *db1, *w2, *b2, *d2, *db2;
  /* Inputs to the function*/
  mxArray const *net, *x, *covstrct;
  
  char *pc_covstrct;
  
  int n = 0, i = 0, i2 = 0, h = 0, h2 = 0, p = 0, p2 = 0, k = 0, 
    ndata = 0, nin = 0, nhidden = 0, nout = 0, tnhidden = 0, tnin = 0, 
    npars = 0, nwts =0, quadexp = 1, dims[3], mark = 0, index = 0, index2 = 0,
    nnonbiasw1 = 0, nnonbiasw2 = 0, ans = 0, strlen = 0,
    covstrct_label = 2;

  int *negativenodes;
  
  double const ERFFACTOR = sqrt(2)/2;
  
  /* Dynamically allocated arrays */    
  double **Theta, *Thetag,  *xTu, *Phi, *sqrtPhi, *aTa, **x2d1, 
    *varequiv, *sdequiv, *aTasquared,
    *gvar, *g2, *g2dashbase, *g, *gdashbase,  
    *gdash_b1, **gdash_w1, *gdash_db1, **gdash_dw1,
    *g2dash_b1, **g2dash_w1, *g2dash_db1, **g2dash_dw1;
  
  double  a0 = 0, gTv = 0, gvardash_u = 0, gvardash_d = 0,
    tempfactor = 0, argument = 0;
  double *pd_qg, *pd_sg, *pd_w1, *pd_b1, *pd_d1, *pd_db1, 
    *pd_w2, *pd_b2, *pd_d2, *pd_db2, *pd_x,
    *pd_gsw1, *pd_gsb1, *pd_gsd1, *pd_gsdb1,
    *pd_gsw2, *pd_gsb2, *pd_gsd2, *pd_gsdb2,
    *pd_gqw1, *pd_gqb1, *pd_gqd1, *pd_gqdb1,
    *pd_gqw2, *pd_gqb2, *pd_gqd2, *pd_gqdb2,
    *pd_nout, *pd_nin, *pd_nhidden;

  /*Make some type checking here*/
  net = prhs[0];
  pd_nout = mxGetPr(mxGetField(net, 0, "nout"));
  pd_nhidden = mxGetPr(mxGetField(net, 0, "nhidden"));
  pd_nin = mxGetPr(mxGetField(net, 0, "nin"));
  nout = (int)pd_nout[0];
  nhidden = (int)pd_nhidden[0];
  nin = (int)pd_nin[0];
  
  tnhidden = nhidden + 1;
  tnin = nin + 1;
  nnonbiasw1 = nin*nhidden;
  nnonbiasw2 = nhidden*nout;
  nwts = tnin*nhidden+tnhidden*nout; 
  npars = nwts;
  covstrct = mxGetField(net, 0, "covstrct");
  strlen = mxGetN(covstrct)+1;
  pc_covstrct = (char *)mxCalloc(strlen, sizeof(char));
  mxGetString(covstrct, pc_covstrct, strlen);
	/* assign a label to covstct according to its value  */
  ans =  strcmp(pc_covstrct, "none");
  if(ans==0)
    covstrct_label = NONE;
  else {
    ans =  strcmp(pc_covstrct, "diag");
    if(ans==0)
      covstrct_label = DIAG;
  
  else
    covstrct_label = OTHER;
  }
  w1 = mxGetField(net, 0, "w1");
  pd_w1 = (double *)mxGetPr(w1);
  b1 = mxGetField(net, 0, "b1");
  pd_b1 = (double *)mxGetPr(b1);
  w2 = mxGetField(net, 0, "w2");
  pd_w2 = (double *)mxGetPr(w2);
  b2 = mxGetField(net, 0, "b2");
  pd_b2 = (double *)mxGetPr(b2);

  switch(covstrct_label){

  case NONE:
    /* do nothing */
  break;	
  case DIAG:{
    npars*=2;
    d1 = mxGetField(net, 0, "d1");
    pd_d1 = (double *)mxGetPr(d1);
    db1 =  mxGetField(net, 0, "db1");
    pd_db1 = (double *)mxGetPr(db1);
    d2 = mxGetField(net, 0, "d2");
    pd_d2 = (double *)mxGetPr(d2);
    db2 = mxGetField(net, 0, "db2");
    pd_db2 = (double *)mxGetPr(db2);
    break;
  }
  default:                      /* an unknown covariance function */
    mexErrMsgTxt("Covariance function not yet implemented");
  }
  
  x = prhs[1];
  pd_x = (double *)mxGetPr(x);
  ndata = (int)mxGetM(x);
 

  if(nrhs < 2)
    quadexp = 0;

  /************************************************************
  Allocate up some general parameters for all expectations
  *************************************************************/               
  dims[0] = nout;
  dims[1] = npars;
  dims[2] = ndata;
  plhs[0] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
  pd_sg = mxGetPr(plhs[0]);

  pd_gsw1 = (double *)mxCalloc(nin*nhidden, sizeof(double));
  pd_gsb1 = (double *)mxCalloc(nhidden, sizeof(double));
  pd_gsw2 = (double *)mxCalloc(nhidden*nout, sizeof(double));
  pd_gsb2 = (double *)mxCalloc(nout, sizeof(double));
  
  g = (double *)mxCalloc(nhidden, sizeof(double));
  gdashbase = (double *)mxCalloc(nhidden, sizeof(double));
  
  aTa = (double *)mxCalloc(nhidden, sizeof(double));
  xTu = (double *)mxCalloc(nhidden, sizeof(double));
  Phi = (double *)mxCalloc(nhidden, sizeof(double));
  sqrtPhi = (double *)mxCalloc(nhidden, sizeof(double));
  
  gdash_b1 = (double *)mxCalloc(nhidden, sizeof(double));
  gdash_w1 = (double **)mxCalloc(nin, sizeof(double *));
  for(i=0; i<nin; i++)
    gdash_w1[i] = (double *)mxCalloc(nhidden, sizeof(double));

  switch(covstrct_label){

  case NONE:
    /* do nothing */
    break;
  default :
    pd_gsd1 = (double *)mxCalloc(nin*nhidden, sizeof(double));
    pd_gsdb1 = (double *)mxCalloc(nhidden, sizeof(double));
    pd_gsd2 = (double *)mxCalloc(nhidden*nout, sizeof(double));
    pd_gsdb2 = (double *)mxCalloc(nout, sizeof(double));
    x2d1 = (double **)mxCalloc(nin, sizeof(double *));
    for(i=0; i<nin; i++)
      x2d1[i] = (double *)mxCalloc(nhidden, sizeof(double));
    gdash_db1 = (double *)mxCalloc(nhidden, sizeof(double));
    gdash_dw1 = (double **)mxCalloc(nin, sizeof(double *));
    for(i=0; i<nin; i++)
      gdash_dw1[i] = (double *)mxCalloc(nhidden, sizeof(double));
   
  }
  /* ***********************************************************
     Set up some general parameters for quadratic expectations
  *************************************************************/               
 
  if(quadexp){   /*If quadratic expectation gradient required */
    plhs[1] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
    pd_qg = mxGetPr(plhs[1]);
    pd_gqw1 = (double *)mxCalloc(nin*nhidden, sizeof(double));
    pd_gqb1 = (double *)mxCalloc(nhidden, sizeof(double));
    pd_gqw2 = (double *)mxCalloc(nhidden*nout, sizeof(double));
    pd_gqb2 = (double *)mxCalloc(nout, sizeof(double));
    

    negativenodes = (int *)mxCalloc(nhidden, sizeof(int));
    Theta = (double **)mxCalloc(tnhidden, sizeof(double *));
    for(h=0; h<tnhidden; h++)
      Theta[h] = (double *)mxCalloc(tnhidden, sizeof(double));
    Thetag = (double *)mxCalloc(tnhidden, sizeof(double));       
    g2 = (double *)mxCalloc(nhidden, sizeof(double));
    g2dash_b1 = (double *)mxCalloc(nhidden, sizeof(double));
    g2dash_w1 = (double **)mxCalloc(nin, sizeof(double *));
    for(i=0; i<nin; i++)
      g2dash_w1[i] = (double *)mxCalloc(nhidden, sizeof(double));
    
    switch(covstrct_label){
      
    case NONE:
      /* do nothing */
      break;
    default:
      pd_gqd1 = (double *)mxCalloc(nin*nhidden, sizeof(double));
      pd_gqdb1 = (double *)mxCalloc(nhidden, sizeof(double));
      pd_gqd2 = (double *)mxCalloc(nhidden*nout, sizeof(double));
      pd_gqdb2 = (double *)mxCalloc(nout, sizeof(double));
      g2dash_db1 = (double *)mxCalloc(nhidden, sizeof(double));
      g2dash_dw1 = (double **)mxCalloc(nin, sizeof(double *));
      for(i=0; i<nin; i++)
        g2dash_dw1[i] = (double *)mxCalloc(nhidden, sizeof(double));
      g2dashbase = (double *)mxCalloc(nhidden, sizeof(double *));  
      gvar = (double *)mxCalloc(nhidden, sizeof(double *));
      varequiv = (double *)mxCalloc(nhidden, sizeof(double *));
      sdequiv = (double *)mxCalloc(nhidden, sizeof(double *));
      aTasquared = (double *)mxCalloc(nhidden, sizeof(double *));
      
    }
  }   
  /* ***********************************************************
     Cycle through the data points and output nodes
  *************************************************************/               
  
  for(n=0; n<ndata; n++){
    for(p=0; p<nout; p++){
      
      /* **************************************************
         Clear all the vectors which store the gradients 
      ***************************************************/
      for(h=0; h<nhidden; h++){
        for(i=0; i<nin;i++)
          pd_gsw1[i+h*nin] = 0.0;         
        pd_gsb1[h] = 0.0;
      }
      for(p2=0; p2<nout; p2++){
        for(h=0; h<nhidden; h++)
          pd_gsw2[h+p2*nhidden] = 0.0;
        pd_gsb2[p] = 0.0;
      }  
      switch(covstrct_label){
      case NONE:      /* no covariance matrix */
        /* do nothing */
        break;
      default:      /* other covariance matrices */
        for(h=0; h<nhidden; h++){
          for(i=0; i<nin;i++)
            pd_gsd1[i+h*nin] = 0.0;         
          pd_gsdb1[h] = 0.0;
        }
        for(p2=0; p2<nout; p2++){
          for(h=0; h<nhidden; h++)
            pd_gsd2[h+p2*nhidden] = 0.0;
          pd_gsdb2[p] = 0.0;
        }  
      }
      
      if(quadexp){
        for(h=0; h<nhidden; h++)
          negativenodes[h] = 0;
        for(h=0; h<nhidden; h++){
          for(i=0; i<nin;i++)
            pd_gqw1[i+h*nin] = 0.0;         
          pd_gqb1[h] = 0.0;
        }
        for(p2=0; p2<nout; p2++){
          for(h=0; h<nhidden; h++)
            pd_gqw2[h+p2*nhidden] = 0.0;
          pd_gqb2[p] = 0.0;
        }  
        switch(covstrct_label){
        case NONE:      /* no covariance matrix */
          /* do nothing */
          break;
        default:      /* other covariance matrices */
          for(h=0; h<nhidden; h++){
            for(i=0; i<nin;i++)
              pd_gqd1[i+h*nin] = 0.0;         
            pd_gqdb1[h] = 0.0;
          }
          for(p2=0; p2<nout; p2++){
            for(h=0; h<nhidden; h++)
              pd_gqd2[h+p2*nhidden] = 0.0;
            pd_gqdb2[p] = 0.0;
          }  
        }
        
      }
      
      /* *******************************************************
         Take the transpose of the data with first layer weights
      ********************************************************/
      for(h=0; h<nhidden; h++){
        xTu[h] = 0;
        for(i=0; i<nin; i++){
          xTu[h] += pd_x[n+i*ndata]*pd_w1[i+h*nin];
        }
        xTu[h] += pd_b1[h];
      }
       /* *******************************************************
         If quadratic expectation is required, fill some matrices
      ********************************************************/
     
      if(quadexp){
        switch(covstrct_label){
        case NONE:      /* no covariance matrix */
          for(h=0; h<nhidden; h++){
            for(h2=h; h2<nhidden; h2++){
              Theta[h][h2] = pd_w2[h+p*nhidden]*pd_w2[h2+p*nhidden];
              Theta[h2][h] = Theta[h][h2];                
            }
            Theta[h][nhidden] = pd_w2[h+p*nhidden]*pd_b2[p];
            Theta[nhidden][h] = Theta[h][nhidden];
          }
          Theta[nhidden][nhidden] = SQUARE(pd_b2[p]);
          break;
        case DIAG:      /* diagonal covariance matrix */
          for(h=0; h<nhidden; h++){
            for(h2=h; h2<nhidden; h2++){
              Theta[h][h2] = pd_w2[h+p*nhidden]*pd_w2[h2+p*nhidden];
              if(h==h2)
                Theta[h][h] += SQUARE(pd_d2[h+p*nhidden]);
              else
                Theta[h2][h] = Theta[h][h2];                
            }
            Theta[h][nhidden] = pd_w2[h+p*nhidden]*pd_b2[p];
            Theta[nhidden][h] = Theta[h][nhidden];
          }
          Theta[nhidden][nhidden] = SQUARE(pd_b2[p])+SQUARE(pd_db2[p]);
          break;
        default:                      /* an unknown covariance function */
          mexErrMsgTxt("Covariance function not yet implemented");
        }
      }
      /* *******************************************************
         Calculate some expectations and useful factors
      ********************************************************/
      for(h=0; h<nhidden; h++){
        a0 = xTu[h];
        switch(covstrct_label){
	
        case NONE:  /* no covariance matrix */
          g[h] = erf(ERFFACTOR*xTu[h]);
          Phi[h] = 1;
          sqrtPhi[h] = 1;
          gdashbase[h] = sqrt(2/PI)*exp(-(SQUARE(a0))/2);
          
          break;
        
        case DIAG:  /* diagonal covariance matrix */
          aTa[h] = 0;
          for(i=0; i<nin; i++)
            aTa[h] += pd_x[n+i*ndata]*pd_x[n+i*ndata]
              *pd_d1[i+h*nin]*pd_d1[i+h*nin];
          aTa[h] += pd_db1[h]*pd_db1[h];
          Phi[h] = 1+aTa[h];
          sqrtPhi[h] = sqrt(Phi[h]);
          /* The expected value of each hidden node*/
          g[h] = erf(ERFFACTOR*a0/sqrtPhi[h]);
          gdashbase[h] = sqrt(2/PI)*exp(-(SQUARE(a0))/(2*Phi[h]));
          break;
        
        default:                /* an unknown covariance function */
          mexErrMsgTxt("Covariance function not yet implemented");
        }
        if(quadexp){    
          switch(covstrct_label){
          case NONE:
            /* The expected value of each hidden node squared */
            g2[h] = SQUARE(g[h]);
            break;
          default: 
            varequiv[h] = 1 + CONSTC*aTa[h];
            sdequiv[h] = sqrt(varequiv[h]);
            aTasquared[h] = SQUARE(aTa[h]);
            /* The expected value of each hidden node squared */
            if(REAPPROX){
              argument = erf(ERFFACTOR*a0/sdequiv[h]);
              g2[h] = 1/sdequiv[h]*(SQUARE(argument)-1)+1;
            }
            else {
              g2[h] = 1-1/sdequiv[h]*
                exp(-CONSTC*SQUARE(a0)*(1/(2*varequiv[h])));
              g2dashbase[h] = -g2[h] + 1;
            }
          }         
        } 
      }
      /* *******************************************************
      Calculate gradients   
      ********************************************************/
      /* Calculate gradient of hidden units expectation with respect
         to first layer weights and biases */
      for(h=0; h<nhidden; h++){
        gdash_b1[h] = gdashbase[h]/sqrtPhi[h];
        for(i=0; i<nin; i++){ 
          gdash_w1[i][h] = pd_x[n+i*ndata]*gdash_b1[h];
        }
      }

      /* calculate gradient of output expectation with respect to
         first layer weights and biases */
      for(h=0; h<nhidden; h++){
        for(i=0; i<nin; i++){
          pd_gsw1[i+h*nin] = gdash_w1[i][h]*pd_w2[h+nhidden*p];
        }
        pd_gsb1[h] = gdash_b1[h]*pd_w2[h+nhidden*p];
      }
 
      /* calculate gradient of output expectation with respect to
         second layer weights and biases */
      for(h=0; h<nhidden; h++)
        pd_gsw2[h+p*nhidden] = g[h];
      pd_gsb2[p] = 1;
    
      /* check if there are other parameters whose gradients are
         required */
      switch(covstrct_label){

      case NONE:
        /*do nothing*/
        break;

      default:
        /* calculate gradient of hidden layer expectation with respect
          to first layer covariance parameters d */
        for(h=0; h<nhidden; h++){
          /*set to Phi^{-3/2}*gdashbase*xTu */
          gdash_db1[h] = -(xTu[h]*gdashbase[h])/(Phi[h]*sqrtPhi[h]);
          for(i=0; i<nin; i++){
            /*create x^2*d1*/
            x2d1[i][h] = SQUARE(pd_x[n+i*ndata])*pd_d1[i+h*nin]; 
            gdash_dw1[i][h] = gdash_db1[h]*x2d1[i][h];
          }
          gdash_db1[h] *= pd_db1[h];
        }

        /* calculate gradient of output expectation with respect to
           first layer covariance parameters d */ 
        for(h=0; h<nhidden; h++){
          for(i=0; i<nin; i++){ 
            pd_gsd1[i+h*nin] =
              gdash_dw1[i][h]*pd_w2[h+p*nhidden]; 
          } 
          pd_gsdb1[h] = gdash_db1[h]*pd_w2[h+p*nhidden];
        } 
        
        switch(covstrct_label){ 
        case DIAG:
          /* do nothing*/ 
          break; 

        default: 
          mexErrMsgTxt("Covariance function not yet implemented");
        } 
      } 
      /* Now check if the gradients of the
         expectation of the square are also required */ 
      if(quadexp){ 
        gTv = 0;
        for(h=0;h<nhidden; h++) 
          gTv += g[h]*pd_w2[h+p*nhidden]; 
        gTv += pd_b2[p];
        for(h=0; h<nhidden; h++){ 
          Thetag[h] = 0;
          for(h2=0; h2<nhidden; h2++) 
            Thetag[h] += Theta[h2][h]*g[h2];
          Thetag[h] += Theta[nhidden][h];
        } 
        pd_gqb2[p] = 2*gTv; 
        
        for(h=0; h<nhidden; h++) 
          pd_gqw2[h+p*nhidden] = 2*gTv*g[h];

        switch(covstrct_label){ 

        case NONE: 
          /* do nothing */ 
          break; 
        
        default:
          for(h=0; h<nhidden; h++){
            gvar[h] = g2[h] - SQUARE(g[h]); 
            /*Due to the approximation it is possible for gvar to be
              small and -ve */
            if(gvar[h]<0.0){
              gvar[h] = 0.0;
              negativenodes[h] = 1;
            }
            /*Due to the approximation it is possible for gvar to be
              small and -ve gvar = gvar.*(gvar>=0);*/
            /* add contribution of var(g) */
            pd_gqw2[h+p*nhidden] += 2*gvar[h]*pd_w2[h+p*nhidden];
            pd_gqd2[h+p*nhidden] = 2*pd_d2[h+p*nhidden]*g2[h];
          } 
          pd_gqdb2[p] = 2*pd_db2[p]; 
          
          switch(covstrct_label){ 

          case DIAG: /* do nothing*/ 
            break;

          default: 
            mexErrMsgTxt("Covariance function not yet implemented");
          } 
        }
        /* calculate gradient of output expectation squared with
           respect to first layer weights and biases */ 
        for(h=0; h<nhidden; h++){
          for(i=0; i<nin; i++){ 
            pd_gqw1[i+h*nin] = 2*Thetag[h]*gdash_w1[i][h]; 
          }
          pd_gqb1[h] = 2*Thetag[h]*gdash_b1[h]; 
        } 
        
        switch(covstrct_label){ 
        case NONE: 
          /* do nothing */ 
          break; 

        default: 
        /* calculate gradient of the expectation of the hidden layer
           squared with respect to first layer weights and biases */
          for(h=0; h<nhidden; h++){ 
            if(negativenodes[h]){
              g2dash_b1[h] = 2*g[h]*gdash_b1[h];
              for(i=0; i<nin; i++){
                g2dash_w1[i][h] = 2*g[h]*gdash_w1[i][h];            
              }
            }
            else{ 
              if(REAPPROX){
                argument = xTu[h]/sdequiv[h];
                g2dash_b1[h] = 1/varequiv[h]*(2*erf(ERFFACTOR*argument)
                                              *sqrt(2/PI)
                                              *exp(-.5*SQUARE(argument)));
              }
              else{
                g2dash_b1[h] = CONSTC*xTu[h]
                  *g2dashbase[h]/varequiv[h];
              }
              for(i=0; i<nin; i++){
                g2dash_w1[i][h] = pd_x[n+i*ndata]*g2dash_b1[h];
              }
            } 
          } 

        /* Add the gradient of the variance of the output squared to
           the gradient of the squared expectation of the output*/
          for(h=0; h<nhidden; h++){ 
            for(i=0; i<nin; i++){ 
              gvardash_u = g2dash_w1[i][h] - 2*g[h]*gdash_w1[i][h];
              pd_gqw1[i+h*nin] += Theta[h][h]*gvardash_u; 
            } 
            gvardash_u = g2dash_b1[h] - 2*g[h]*gdash_b1[h]; 
            pd_gqb1[h] += Theta[h][h]*gvardash_u; 
          } 
        /* calculate gradient of the expectation of the hidden layer
           squared with respect to first layer covariance parameters d */
          for(h=0; h<nhidden; h++){ 
            if(negativenodes[h]){
              g2dash_db1[h] = 2*g[h]*gdash_db1[h];
              for(i=0; i<nin; i++){
                g2dash_dw1[i][h] = 2*g[h]*gdash_dw1[i][h];            
              }
            }
            else{ 
              if(REAPPROX){
                argument = xTu[h]/sdequiv[h];
                tempfactor = (g2[h]-1) + xTu[h]/varequiv[h]
                  *(2*erf(ERFFACTOR*argument)*sqrt(2/PI)
                    *exp(-.5*SQUARE(argument)));
                tempfactor *= -CONSTC/varequiv[h];
              }
              else{
                tempfactor = CONSTC*
                  (1/varequiv[h] - CONSTC*SQUARE(xTu[h])/SQUARE(varequiv[h]))
                  *g2dashbase[h]; 
              }
              for(i=0; i<nin; i++) 
                g2dash_dw1[i][h] = x2d1[i][h]*tempfactor; 
              g2dash_db1[h] = pd_db1[h]*tempfactor; 
            }
          } 
          /* Add the gradient of the variance of the output squared to
             the gradient of the squared expectation of the output*/ 
          for(h=0; h<nhidden; h++){
            for(i=0; i<nin; i++){ 
              gvardash_d = g2dash_dw1[i][h] - 2*g[h]*gdash_dw1[i][h]; 
              pd_gqd1[i+h*nin] = 2*Thetag[h]*gdash_dw1[i][h];
              pd_gqd1[i+h*nin] += Theta[h][h]*gvardash_d;
            } 
            gvardash_d = g2dash_db1[h] - 2*g[h]*gdash_db1[h];
            pd_gqdb1[h] = 2*Thetag[h]*gdash_db1[h];
            pd_gqdb1[h] += Theta[h][h]*gvardash_d;
          } 
          
          switch(covstrct_label){ 
          
          case DIAG:
            /* do nothing */ 
            break;

          default:
            mexErrMsgTxt("Covariance function not yet implemented");
          }
        }
        /* This routine is equivalent to enspak, it is taking all the
           gradients calculated and packing them into an mxArray which is of
           dimension nout*nparameters*ndata */ 
        index = 0;
        for(index2=0; index2<(nin*nhidden); index2++, index++)
          pd_qg[p+(index+n*npars)*nout] = pd_gqw1[index2];
        for(index2=0; index2<nhidden; index2++, index++)
          pd_qg[p+(index+n*npars)*nout] = pd_gqb1[index2];
        for(index2=0; index2<(nhidden*nout); index2++, index++)
          pd_qg[p+(index+n*npars)*nout] = pd_gqw2[index2];
        for(index2=0; index2<nout; index2++, index++)
          pd_qg[p+(index+n*npars)*nout] = pd_gqb2[index2];
        if(index<npars){
          for(index2=0; index2<(nin*nhidden); index2++, index++)
            pd_qg[p+(index+n*npars)*nout] = pd_gqd1[index2];
          for(index2=0; index2<nhidden; index2++, index++)
            pd_qg[p+(index+n*npars)*nout] = pd_gqdb1[index2];
          for(index2=0; index2<(nhidden*nout); index2++, index++)
            pd_qg[p+(index+n*npars)*nout] = pd_gqd2[index2]; 
          for(index2=0; index2<nout; index2++, index++)
            pd_qg[p+(index+n*npars)*nout] = pd_gqdb2[index2];
        }      
      } 
      index = 0;
      for(index2=0; index2<(nin*nhidden); index2++, index++)
        pd_sg[p+(index+n*npars)*nout] = pd_gsw1[index2];
      for(index2=0; index2<nhidden; index2++, index++)
        pd_sg[p+(index+n*npars)*nout] = pd_gsb1[index2];
      for(index2=0; index2<(nhidden*nout); index2++, index++)
        pd_sg[p+(index+n*npars)*nout] = pd_gsw2[index2]; 
      for(index2=0; index2<nout; index2++, index++)
        pd_sg[p+(index+n*npars)*nout] = pd_gsb2[index2];
      if(index<npars){
        for(index2=0; index2<(nin*nhidden); index2++, index++)
          pd_sg[p+(index+n*npars)*nout] = pd_gsd1[index2];
        for(index2=0; index2<nhidden; index2++, index++)
          pd_sg[p+(index+n*npars)*nout] = pd_gsdb1[index2];
        for(index2=0; index2<(nhidden*nout); index2++, index++)
          pd_sg[p+(index+n*npars)*nout] = pd_gsd2[index2]; 
        for(index2=0; index2<nout; index2++, index++)
          pd_sg[p+(index+n*npars)*nout] = pd_gsdb2[index2];
      } 
    }
  }
  if(quadexp){
    mxFree(pd_gqw1);
    mxFree(pd_gqb1);
    mxFree(pd_gqw2);
    mxFree(pd_gqb2);
    mxFree(negativenodes);
    for(h=0; h<tnhidden; h++)  
      mxFree(Theta[h]);
    mxFree(Theta);
    mxFree(Thetag);
    mxFree(g2);
    
    switch(covstrct_label){
      
    case NONE:
      /* do nothing */
      break;
    
    default:
      mxFree(pd_gqd1);
      mxFree(pd_gqdb1);
      mxFree(pd_gqd2);
      mxFree(pd_gqdb2);
      mxFree(g2dashbase);
      mxFree(varequiv);
      mxFree(gvar);
      mxFree(sdequiv); 
      mxFree(aTasquared);
      for(i=0; i<nin; i++)  
        mxFree(g2dash_w1[i]);
      mxFree(g2dash_w1);
      mxFree(g2dash_b1);
      for(i=0; i<nin; i++)  
        mxFree(g2dash_dw1[i]);
      mxFree(g2dash_dw1);
      mxFree(g2dash_db1);
    }
  }
  mxFree(pd_gsw1);
  mxFree(pd_gsb1);
  mxFree(pd_gsw2);
  mxFree(pd_gsb2);
  mxFree(xTu);
  mxFree(Phi);
  mxFree(sqrtPhi); 
  mxFree(g);
  mxFree(gdashbase);
  mxFree(aTa);
  for(i=0; i<nin; i++)
    mxFree(gdash_w1[i]);
  mxFree(gdash_w1);
  mxFree(gdash_b1);
  
  switch(covstrct_label){
    
  case NONE:
    /* do nothing */
    break;
  
  default:
    mxFree(pd_gsd1);
    mxFree(pd_gsdb1);
    mxFree(pd_gsd2);
    mxFree(pd_gsdb2);
    for(i=0; i<nin; i++)  
      mxFree(gdash_dw1[i]);
    mxFree(gdash_dw1);
    mxFree(gdash_db1);
    for(i=0; i<nin; i++)  
      mxFree(x2d1[i]);
    mxFree(x2d1);
  }
}

double erfcc(double x)
{
  double t, z, ans;
  z = fabs(x);
  t = 1.0/(1.0 + 0.5*z);
  ans = t*exp(-z*z - 1.26551223+t*(1.00002368+t*(0.37409196+t*(0.09678418+
    t*(-0.18628806+t*(0.27886807+t*(-1.13520398+t*(1.48851587+
    t*(-0.82215223+t*0.17087277)))))))));
  return x >= 0.0 ? ans:2.0-ans;
}

double erf(double x)
{
  double gammp(double a, double x);
  mxArray *callplhs[1], *callprhs[1];
  double *pd_y;
  int dims[1];
  if(ERFTYPE == USE_MATLAB){
    dims[0] = 1;
    callprhs[0] = mxCreateNumericArray(1, dims, mxDOUBLE_CLASS, mxREAL);
    mxSetPr(callprhs[0], &x);
    mexCallMATLAB(1, callplhs, 1, callprhs, "erf");
    pd_y = (double *)mxGetPr(callplhs[0]);
    return pd_y[0];
  }
  else if(ERFTYPE == USE_APPROX)
    return 1-erfcc(x);
  
  else if(ERFTYPE == USE_C)
    ;/*return x < 0.0 ? -gammp(0.5, SQUARE(x)) : gammp(0.5, SQUARE(x));*/

}
/*double gammp(double a, double x)
{
  void gcf(double *gammcf, double a, double x, double *gln);
  void gser(double *gamser, double a, double x, double *gln);

  double gamser, gammcf, gln;
 
  if(x < 0.0|| a <= 0.0)
    mexErrMsgTxt("Invalid arguments in routine gammp");
  if(x <(a + 1.0)){
    gser(&gamser, a, x, &gln);
    return gamser;
  } else {
    gcf(&gammcf, a, x, &gln);
    return 1.0-gammcf;
  }
}
double gammq(double a, double x)
{
  void gcf(double *gammcf, double a, double x, double *gln);  
  void gser(double *gamser, double a, double x, double *gln);
  
  double gamser, gammcf, gln;

  if(x < 0.0 || a <= 0.0)
    mexErrMsgTxt("Invalid arguments in routine gammp");
  if(x <(a + 1.0)){
    gser(&gamser, a, x, &gln);
    return 1.0-gamser;
  } 
  else {
    gcf(&gammcf, a, x, &gln);
    return gammcf;
  }
}
void gammaln(double a);
{
  mexErrMsgTxt("gammaln not yet implemented");
}
void gser(double *gamser, double a, double x, double *gln)
{
  double gammln(double xx);
  int n;
  double sum, del, ap;
  
  *gln=gammaln(a);
  if(x <= 0.0){
    if(x <0.0)
      mexErrMsgTxt("Invalid arguments in routine gser");
    *gamser = 0.0;
    return;
  } else {
    ap = a;
    del = sum =1.0/a;
    for (n=1; n<=ITMAX; n++) {
      ++ap;
      del *= x/ap;
      sum += del;
      if (fabs(del) < fabs(sum)*EPS) {
        *gamser = sum*exp(-x+a*log(x)-(*gln));
        return;
      }
    }
    mexErrMsgTxt("a too large, ITMAX too small in routine gser");
    return;
  }
}

void gcf(double *gammcf, double a, double x, double *gln)
  
{
  double gammln(double xx);
  
  int i;
  double an, b, c, d, del, h;
  
  *gln = gammaln(a);
  b = x+1.0-a;
  c = 1.0/FPMIN;
  d = 1.0/b;
  h = d;
  
  for(i=1; i<=ITMAX; i++){
    an = -i*(i-a);
    b += 2.0;
    d = an*d + b;
    if (fabs(d) < FPMIN)
      d = FPMIN;
    c = b+an/c;
    if (fabs(c) < FPMIN)
      c = FPMIN;
    d = 1.0/d;
    del = d*c;
    h *= del;
    if(fabs(del-1.0)<EPS) 
      break;
  }
  if(i>ITMAX)
    mexErrMsgTxt("a too large, ITMAX too small in gcf");
  *gammcf = exp(-x+a*log(c)-(*gln))*h;
}

*/
