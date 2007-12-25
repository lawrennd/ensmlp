#include "ensexpgrad.h"

void mexFunction(
                 int nlhs,       mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]
		 )

{/*function [sg, qg] = ensexpgrad(net, x)*/
  
  /* Inputs to the function*/
  mxArray *pmxTemp;
  
  char *pcTemp;
  
  int n = 0, i = 0, h = 0, h2 = 0, p = 0, k = 0, 
    numData = 0, 
    quadexp = 1, dims[3], mark = 0, index = 0, index2 = 0,
    numNonBiasW1 = 0, numNonBiasW2 = 0, ans = 0, strlen = 0,
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
  double *quadGrad, *singGrad, *x, *pdTemp;
  
  ens net, netSingGrad, netQuadGrad;


  /*Should make some type checking here*/


  pdTemp = mxGetPr(mxGetField(prhs[0], 0, "nout"));
  net.numOut = (int)(*pdTemp);
  pdTemp = mxGetPr(mxGetField(prhs[0], 0, "nhidden"));
  net.numHidden = (int)(*pdTemp);
  pdTemp = mxGetPr(mxGetField(prhs[0], 0, "nin"));
  net.numIn = (int)(*pdTemp);
  
  net.numWeights = (net.numIn + 1)*net.numHidden 
    + (net.numHidden + 1)*net.numOut; 
  net.numParams = net.numWeights;
  
  pmxTemp = mxGetField(prhs[0], 0, "covstrct");
  strlen = mxGetN(pmxTemp)+1;
  pcTemp = (char *)mxCalloc(strlen, sizeof(char));
  mxGetString(pmxTemp, pcTemp, strlen);
	/* assign a label to covstct according to its value  */
  ans =  strcmp(pcTemp, "none");
  if(ans==0)
    net.covarStruct = NONE;
  else {
    ans =  strcmp(pcTemp, "diag");
    if(ans==0)
      net.covarStruct = DIAG;
  else
    net.covarStruct = OTHER;
  }

  pmxTemp = mxGetField(prhs[0], 0, "w1");
  net.w1 = (double *)mxGetPr(pmxTemp);
  pmxTemp = mxGetField(prhs[0], 0, "b1");
  net.b1 = (double *)mxGetPr(pmxTemp);
  pmxTemp = mxGetField(prhs[0], 0, "w2");
  net.w2 = (double *)mxGetPr(pmxTemp);
  pmxTemp = mxGetField(prhs[0], 0, "b2");
  net.b2 = (double *)mxGetPr(pmxTemp);

  switch(net.covarStruct){

  case NONE:
    /* do nothing */
    break;	
  case DIAG:{
    net.numParams*=2;
    pmxTemp = mxGetField(prhs[0], 0, "d1");
    net.d1 = (double *)mxGetPr(pmxTemp);
    pmxTemp =  mxGetField(prhs[0], 0, "db1");
    net.db1 = (double *)mxGetPr(pmxTemp);
    pmxTemp = mxGetField(prhs[0], 0, "d2");
    net.d2 = (double *)mxGetPr(pmxTemp);
    pmxTemp = mxGetField(prhs[0], 0, "db2");
    net.db2 = (double *)mxGetPr(pmxTemp);
    break;
  }
  default:                      /* an unknown covariance function */
    mexErrMsgTxt("Covariance function not yet implemented");
  }
  
  x = (double *)mxGetPr(prhs[1]);
  numData = (int)mxGetM(prhs[1]);
 

  if(nrhs < 2)
    quadexp = 0;

  /************************************************************
  Allocate up some general parameters for all expectations
  *************************************************************/               
  dims[0] = net.numOut;
  dims[1] = net.numParams;
  dims[2] = numData;
  plhs[0] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
  singGrad = mxGetPr(plhs[0]);

  ens_initialise(&netSingGrad, net.numIn, net.numHidden, 
                 net.numOut, net.covarStruct);
  
  g = (double *)mxCalloc(net.numHidden, sizeof(double));
  gdashbase = (double *)mxCalloc(net.numHidden, sizeof(double));
  
  aTa = (double *)mxCalloc(net.numHidden, sizeof(double));
  xTu = (double *)mxCalloc(net.numHidden, sizeof(double));
  Phi = (double *)mxCalloc(net.numHidden, sizeof(double));
  sqrtPhi = (double *)mxCalloc(net.numHidden, sizeof(double));
  
  gdash_b1 = (double *)mxCalloc(net.numHidden, sizeof(double));
  gdash_w1 = (double **)mxCalloc(net.numIn, sizeof(double *));
  for(i=0; i<net.numIn; i++)
    gdash_w1[i] = (double *)mxCalloc(net.numHidden, sizeof(double));

  switch(net.covarStruct){

  case NONE:
    /* do nothing */
    break;
  default :
    x2d1 = (double **)mxCalloc(net.numIn, sizeof(double *));
    for(i=0; i<net.numIn; i++)
      x2d1[i] = (double *)mxCalloc(net.numHidden, sizeof(double));
    gdash_db1 = (double *)mxCalloc(net.numHidden, sizeof(double));
    gdash_dw1 = (double **)mxCalloc(net.numIn, sizeof(double *));
    for(i=0; i<net.numIn; i++)
      gdash_dw1[i] = (double *)mxCalloc(net.numHidden, sizeof(double));
   
  }
  /* ***********************************************************
     Set up some general parameters for quadratic expectations
  *************************************************************/               
 
  if(quadexp){   /*If quadratic expectation gradient required */

    plhs[1] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
    quadGrad = mxGetPr(plhs[1]);

    ens_initialise(&netQuadGrad, net.numIn, net.numHidden, 
                   net.numOut, net.covarStruct);    

    negativenodes = (int *)mxCalloc(net.numHidden, sizeof(int));
    Theta = (double **)mxCalloc(net.numHidden + 1, sizeof(double *));
    for(h=0; h<(net.numHidden + 1); h++)
      Theta[h] = (double *)mxCalloc((net.numHidden + 1), sizeof(double));
    Thetag = (double *)mxCalloc(net.numHidden + 1, sizeof(double));       
    g2 = (double *)mxCalloc(net.numHidden, sizeof(double));
    g2dash_b1 = (double *)mxCalloc(net.numHidden, sizeof(double));
    g2dash_w1 = (double **)mxCalloc(net.numIn, sizeof(double *));
    for(i=0; i<net.numIn; i++)
      g2dash_w1[i] = (double *)mxCalloc(net.numHidden, sizeof(double));
    
    switch(net.covarStruct){
      
    case NONE:
      /* do nothing */
      break;

    default:
      g2dash_db1 = (double *)mxCalloc(net.numHidden, sizeof(double));
      g2dash_dw1 = (double **)mxCalloc(net.numIn, sizeof(double *));
      for(i=0; i<net.numIn; i++)
        g2dash_dw1[i] = (double *)mxCalloc(net.numHidden, sizeof(double));
      g2dashbase = (double *)mxCalloc(net.numHidden, sizeof(double *));  
      gvar = (double *)mxCalloc(net.numHidden, sizeof(double *));
      varequiv = (double *)mxCalloc(net.numHidden, sizeof(double *));
      sdequiv = (double *)mxCalloc(net.numHidden, sizeof(double *));
      aTasquared = (double *)mxCalloc(net.numHidden, sizeof(double *));
      
    }
  }   
  /* ***********************************************************
     Cycle through the data points and output nodes
  *************************************************************/               
  
  for(n=0; n<numData; n++){
    for(p=0; p<net.numOut; p++){
      
      ens_zeroParameters(&netSingGrad);
      if(quadexp)
        ens_zeroParameters(&netQuadGrad);   
      
      /* *******************************************************
         Take the transpose of the data with first layer weights
      ********************************************************/
      for(h=0; h<net.numHidden; h++){
        xTu[h] = 0;
        for(i=0; i<net.numIn; i++){
          xTu[h] += x[n+i*numData]*net.w1[i+h*net.numIn];
        }
        xTu[h] += net.b1[h];
      }
       /* *******************************************************
         If quadratic expectation is required, fill some matrices
      ********************************************************/
     
      if(quadexp){
        switch(net.covarStruct){
        case NONE:      /* no covariance matrix */
          for(h=0; h<net.numHidden; h++){
            for(h2=h; h2<net.numHidden; h2++){
              Theta[h][h2] = net.w2[h+p*net.numHidden]
                *net.w2[h2+p*net.numHidden];
              Theta[h2][h] = Theta[h][h2];                
            }
            Theta[h][net.numHidden] = net.w2[h+p*net.numHidden]*net.b2[p];
            Theta[net.numHidden][h] = Theta[h][net.numHidden];
          }
          Theta[net.numHidden][net.numHidden] = SQUARE(net.b2[p]);
          break;
        case DIAG:      /* diagonal covariance matrix */
          for(h=0; h<net.numHidden; h++){
            for(h2=h; h2<net.numHidden; h2++){
              Theta[h][h2] = net.w2[h+p*net.numHidden]
                *net.w2[h2+p*net.numHidden];
              if(h==h2)
                Theta[h][h] += SQUARE(net.d2[h+p*net.numHidden]);
              else
                Theta[h2][h] = Theta[h][h2];                
            }
            Theta[h][net.numHidden] = net.w2[h+p*net.numHidden]*net.b2[p];
            Theta[net.numHidden][h] = Theta[h][net.numHidden];
          }
          Theta[net.numHidden][net.numHidden] = SQUARE(net.b2[p])
            +SQUARE(net.db2[p]);
          break;
        default:                      /* an unknown covariance function */
          mexErrMsgTxt("Covariance function not yet implemented");
        }
      }
      /* *******************************************************
         Calculate some expectations and useful factors
      ********************************************************/
      for(h=0; h<net.numHidden; h++){
        a0 = xTu[h];
        switch(net.covarStruct){
	
        case NONE:  /* no covariance matrix */
          g[h] = erf(ERFFACTOR*xTu[h]);
          Phi[h] = 1;
          sqrtPhi[h] = 1;
          gdashbase[h] = sqrt(2/PI)*exp(-(SQUARE(a0))/2);
          
          break;
        
        case DIAG:  /* diagonal covariance matrix */
          aTa[h] = 0;
          for(i=0; i<net.numIn; i++)
            aTa[h] += x[n+i*numData]*x[n+i*numData]
              *net.d1[i+h*net.numIn]*net.d1[i+h*net.numIn];
          aTa[h] += net.db1[h]*net.db1[h];
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
          switch(net.covarStruct){
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
      for(h=0; h<net.numHidden; h++){
        gdash_b1[h] = gdashbase[h]/sqrtPhi[h];
        for(i=0; i<net.numIn; i++){ 
          gdash_w1[i][h] = x[n+i*numData]*gdash_b1[h];
        }
      }

      /* calculate gradient of output expectation with respect to
         first layer weights and biases */
      for(h=0; h<net.numHidden; h++){
        for(i=0; i<net.numIn; i++){
          netSingGrad.w1[i+h*net.numIn] = gdash_w1[i][h]
            *net.w2[h+net.numHidden*p];
        }
        netSingGrad.b1[h] = gdash_b1[h]*net.w2[h+net.numHidden*p];
      }
 
      /* calculate gradient of output expectation with respect to
         second layer weights and biases */
      for(h=0; h<net.numHidden; h++)
        netSingGrad.w2[h+p*net.numHidden] = g[h];
      netSingGrad.b2[p] = 1;
    
      /* check if there are other parameters whose gradients are
         required */
      switch(net.covarStruct){

      case NONE:
        /*do nothing*/
        break;

      default:
        /* calculate gradient of hidden layer expectation with respect
          to first layer covariance parameters d */
        for(h=0; h<net.numHidden; h++){
          /*set to Phi^{-3/2}*gdashbase*xTu */
          gdash_db1[h] = -(xTu[h]*gdashbase[h])/(Phi[h]*sqrtPhi[h]);
          for(i=0; i<net.numIn; i++){
            /*create x^2*d1*/
            x2d1[i][h] = SQUARE(x[n+i*numData])*net.d1[i+h*net.numIn]; 
            gdash_dw1[i][h] = gdash_db1[h]*x2d1[i][h];
          }
          gdash_db1[h] *= net.db1[h];
        }

        /* calculate gradient of output expectation with respect to
           first layer covariance parameters d */ 
        for(h=0; h<net.numHidden; h++){
          for(i=0; i<net.numIn; i++){ 
            netSingGrad.d1[i+h*net.numIn] =
              gdash_dw1[i][h]*net.w2[h+p*net.numHidden]; 
          } 
          netSingGrad.db1[h] = gdash_db1[h]*net.w2[h+p*net.numHidden];
        } 
        
        switch(net.covarStruct){ 
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
        for(h=0;h<net.numHidden; h++) 
          gTv += g[h]*net.w2[h+p*net.numHidden]; 
        gTv += net.b2[p];
        for(h=0; h<net.numHidden; h++){ 
          Thetag[h] = 0;
          for(h2=0; h2<net.numHidden; h2++) 
            Thetag[h] += Theta[h2][h]*g[h2];
          Thetag[h] += Theta[net.numHidden][h];
        } 
        netQuadGrad.b2[p] = 2*gTv; 
        
        for(h=0; h<net.numHidden; h++) 
          netQuadGrad.w2[h+p*net.numHidden] = 2*gTv*g[h];

        switch(net.covarStruct){ 

        case NONE: 
          /* do nothing */ 
          break; 
        
        default:
          for(h=0; h<net.numHidden; h++){
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
            netQuadGrad.w2[h+p*net.numHidden] += 2*gvar[h]*net.w2[h+p*net.numHidden];
            netQuadGrad.d2[h+p*net.numHidden] = 2*net.d2[h+p*net.numHidden]*g2[h];
          } 
          netQuadGrad.db2[p] = 2*net.db2[p]; 
          
          switch(net.covarStruct){ 

          case DIAG: /* do nothing*/ 
            break;

          default: 
            mexErrMsgTxt("Covariance function not yet implemented");
          } 
        }
        /* calculate gradient of output expectation squared with
           respect to first layer weights and biases */ 
        for(h=0; h<net.numHidden; h++){
          for(i=0; i<net.numIn; i++){ 
            netQuadGrad.w1[i+h*net.numIn] = 2*Thetag[h]*gdash_w1[i][h]; 
          }
          netQuadGrad.b1[h] = 2*Thetag[h]*gdash_b1[h]; 
        } 
        
        switch(net.covarStruct){ 
        case NONE: 
          /* do nothing */ 
          break; 

        default: 
        /* calculate gradient of the expectation of the hidden layer
           squared with respect to first layer weights and biases */
          for(h=0; h<net.numHidden; h++){ 
            if(negativenodes[h]){
              g2dash_b1[h] = 2*g[h]*gdash_b1[h];
              for(i=0; i<net.numIn; i++){
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
              for(i=0; i<net.numIn; i++){
                g2dash_w1[i][h] = x[n+i*numData]*g2dash_b1[h];
              }
            } 
          } 

        /* Add the gradient of the variance of the output squared to
           the gradient of the squared expectation of the output*/
          for(h=0; h<net.numHidden; h++){ 
            for(i=0; i<net.numIn; i++){ 
              gvardash_u = g2dash_w1[i][h] - 2*g[h]*gdash_w1[i][h];
              netQuadGrad.w1[i+h*net.numIn] += Theta[h][h]*gvardash_u; 
            } 
            gvardash_u = g2dash_b1[h] - 2*g[h]*gdash_b1[h]; 
            netQuadGrad.b1[h] += Theta[h][h]*gvardash_u; 
          } 
        /* calculate gradient of the expectation of the hidden layer
           squared with respect to first layer covariance parameters d */
          for(h=0; h<net.numHidden; h++){ 
            if(negativenodes[h]){
              g2dash_db1[h] = 2*g[h]*gdash_db1[h];
              for(i=0; i<net.numIn; i++){
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
              for(i=0; i<net.numIn; i++) 
                g2dash_dw1[i][h] = x2d1[i][h]*tempfactor; 
              g2dash_db1[h] = net.db1[h]*tempfactor; 
            }
          } 
          /* Add the gradient of the variance of the output squared to
             the gradient of the squared expectation of the output*/ 
          for(h=0; h<net.numHidden; h++){
            for(i=0; i<net.numIn; i++){ 
              gvardash_d = g2dash_dw1[i][h] - 2*g[h]*gdash_dw1[i][h]; 
              netQuadGrad.d1[i+h*net.numIn] = 2*Thetag[h]*gdash_dw1[i][h];
              netQuadGrad.d1[i+h*net.numIn] += Theta[h][h]*gvardash_d;
            } 
            gvardash_d = g2dash_db1[h] - 2*g[h]*gdash_db1[h];
            netQuadGrad.db1[h] = 2*Thetag[h]*gdash_db1[h];
            netQuadGrad.db1[h] += Theta[h][h]*gvardash_d;
          } 
          
          switch(net.covarStruct){ 
          
          case DIAG:
            /* do nothing */ 
            break;

          default:
            mexErrMsgTxt("Covariance function not yet implemented");
          }
        }
        /* This routine is equivalent to enspak, it is taking all the
           gradients calculated and packing them into an mxArray which is of
           dimension net.numOut*net.numParams*numData */ 
        index = 0;
        for(index2=0; index2<(net.numIn*net.numHidden); index2++, index++)
          quadGrad[p+(index+n*net.numParams)*net.numOut] = netQuadGrad.w1[index2];
        for(index2=0; index2<net.numHidden; index2++, index++)
          quadGrad[p+(index+n*net.numParams)*net.numOut] = netQuadGrad.b1[index2];
        for(index2=0; index2<(net.numHidden*net.numOut); index2++, index++)
          quadGrad[p+(index+n*net.numParams)*net.numOut] = netQuadGrad.w2[index2];
        for(index2=0; index2<net.numOut; index2++, index++)
          quadGrad[p+(index+n*net.numParams)*net.numOut] = netQuadGrad.b2[index2];
        if(index<net.numParams){
          for(index2=0; index2<(net.numIn*net.numHidden); index2++, index++)
            quadGrad[p+(index+n*net.numParams)*net.numOut] = netQuadGrad.d1[index2];
          for(index2=0; index2<net.numHidden; index2++, index++)
            quadGrad[p+(index+n*net.numParams)*net.numOut] = netQuadGrad.db1[index2];
          for(index2=0; index2<(net.numHidden*net.numOut); index2++, index++)
            quadGrad[p+(index+n*net.numParams)*net.numOut] = netQuadGrad.d2[index2]; 
          for(index2=0; index2<net.numOut; index2++, index++)
            quadGrad[p+(index+n*net.numParams)*net.numOut] = netQuadGrad.db2[index2];
        }      
      } 
      index = 0;
      for(index2=0; index2<(net.numIn*net.numHidden); index2++, index++)
        singGrad[p+(index+n*net.numParams)*net.numOut] = netSingGrad.w1[index2];
      for(index2=0; index2<net.numHidden; index2++, index++)
        singGrad[p+(index+n*net.numParams)*net.numOut] = netSingGrad.b1[index2];
      for(index2=0; index2<(net.numHidden*net.numOut); index2++, index++)
        singGrad[p+(index+n*net.numParams)*net.numOut] = netSingGrad.w2[index2]; 
      for(index2=0; index2<net.numOut; index2++, index++)
        singGrad[p+(index+n*net.numParams)*net.numOut] = netSingGrad.b2[index2];
      if(index<net.numParams){
        for(index2=0; index2<(net.numIn*net.numHidden); index2++, index++)
          singGrad[p+(index+n*net.numParams)*net.numOut] = netSingGrad.d1[index2];
        for(index2=0; index2<net.numHidden; index2++, index++)
          singGrad[p+(index+n*net.numParams)*net.numOut] = netSingGrad.db1[index2];
        for(index2=0; index2<(net.numHidden*net.numOut); index2++, index++)
          singGrad[p+(index+n*net.numParams)*net.numOut] = netSingGrad.d2[index2]; 
        for(index2=0; index2<net.numOut; index2++, index++)
          singGrad[p+(index+n*net.numParams)*net.numOut] = netSingGrad.db2[index2];
      } 
    }
  }
  if(quadexp){
    mxFree(netQuadGrad.w1);
    mxFree(netQuadGrad.b1);
    mxFree(netQuadGrad.w2);
    mxFree(netQuadGrad.b2);
    mxFree(negativenodes);
    for(h=0; h<=net.numHidden; h++)  
      mxFree(Theta[h]);
    mxFree(Theta);
    mxFree(Thetag);
    mxFree(g2);
    
    switch(net.covarStruct){
      
    case NONE:
      /* do nothing */
      break;
    
    default:
      mxFree(netQuadGrad.d1);
      mxFree(netQuadGrad.db1);
      mxFree(netQuadGrad.d2);
      /*mxFree(netQuadGrad.db2);*/
      mxFree(g2dashbase);
      mxFree(varequiv);
      mxFree(gvar);
      mxFree(sdequiv); 
      mxFree(aTasquared);
      for(i=0; i<net.numIn; i++)  
        mxFree(g2dash_w1[i]);
      mxFree(g2dash_w1);
      mxFree(g2dash_b1);
      for(i=0; i<net.numIn; i++)  
        mxFree(g2dash_dw1[i]);
      mxFree(g2dash_dw1);
      mxFree(g2dash_db1);
    }
  }
  mxFree(netSingGrad.w1);
  mxFree(netSingGrad.b1);
  mxFree(netSingGrad.w2);
  mxFree(netSingGrad.b2);
  mxFree(xTu);
  mxFree(Phi);
  mxFree(sqrtPhi); 
  mxFree(g);
  mxFree(gdashbase);
  mxFree(aTa);
  for(i=0; i<net.numIn; i++)
    mxFree(gdash_w1[i]);
  mxFree(gdash_w1);
  mxFree(gdash_b1);
  
  switch(net.covarStruct){
    
  case NONE:
    /* do nothing */
    break;
  
  default:
    mxFree(netSingGrad.d1);
    mxFree(netSingGrad.db1);
    mxFree(netSingGrad.d2);
    mxFree(netSingGrad.db2);
    for(i=0; i<net.numIn; i++)  
      mxFree(gdash_dw1[i]);
    mxFree(gdash_dw1);
    mxFree(gdash_db1);
    for(i=0; i<net.numIn; i++)  
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

void ens_zeroParameters(ens *net)
{
  int h = 0, i = 0, p = 0;
  for(h=0; h<net->numHidden; h++){
    for(i=0; i<net->numIn;i++)
      net->w1[i+h*net->numIn] = 0.0;         
    net->b1[h] = 0.0;
  }
  for(p=0; p<net->numOut; p++){
    for(h=0; h<net->numHidden; h++)
      net->w2[h+p*net->numHidden] = 0.0;
    net->b2[p] = 0.0;
  }  
  switch(net->covarStruct){
  case NONE:      /* no covariance matrix */
    /* do nothing */
    break;
  default:      /* other covariance matrices */
    for(h=0; h<net->numHidden; h++){
      for(i=0; i<net->numIn;i++)
        net->d1[i+h*net->numIn] = 0.0;         
      net->db1[h] = 0.0;
    }
    for(p=0; p<net->numOut; p++){
      for(h=0; h<net->numHidden; h++)
        net->d2[h+p*net->numHidden] = 0.0;
      net->db2[p] = 0.0;
    }  
  }
}
void ens_initialise(ens *net, 
                    int numIn, 
                    int numHidden, 
                    int numOut, 
                    int covarStruct)
{
  net->numIn = numIn;
  net->numHidden = numHidden;
  net->numOut = numOut;

  net->numWeights = (net->numIn + 1)*net->numHidden 
    + (net->numHidden + 1)*net->numOut; 
  net->numParams = net->numWeights;

  net->w1 = (double *)mxCalloc(numIn*numHidden, sizeof(double));
  net->b1 = (double *)mxCalloc(numHidden, sizeof(double));
  net->w2 = (double *)mxCalloc(numHidden*numOut, sizeof(double));
  net->b2 = (double *)mxCalloc(numOut, sizeof(double));
  
  switch(net->covarStruct){

  case NONE:
    /* do nothing */
    break;
  default :
    net->numParams *= 2;
    net->d1 = (double *)mxCalloc(net->numIn*net->numHidden, sizeof(double));
    net->db1 = (double *)mxCalloc(net->numHidden, sizeof(double));
    net->d2 = (double *)mxCalloc(net->numHidden*net->numOut, sizeof(double));
    net->db2 = (double *)mxCalloc(net->numOut, sizeof(double));
  }
}
