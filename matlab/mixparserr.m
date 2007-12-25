function [e, E_D, eprior, entropy] = mixparserr(net, x, t)

% MIXPARSERR Portion of error function associated with mixture parameters.
% FORMAT
% DESC takes the network structure from a mixture of ensembles and returns
% the error function.
% ARG net : network for which likelihood is required.
% ARG x : input data.
% ARG t : target data.
% RETURN e : the error function.
%
% DESC also returns separately the data, entropy and prior contributions to
% the error function. In the case of multiple groups in the prior, GPRIOR is a
% matrix with a row for each group and a column for each weight parameter.
% ARG net : network for which likelihood is required.
% ARG x : input data.
% ARG t : target data.
% RETURN g : lower bound on log likelihood.
% RETURN E : total error.
% RETURN EPRIOR : prior error.
% RETURN ENTROPY : entropy error.
%
% SEEALSO : mixens, mixparsgrad, mixparsunpak, mixparspak
%
% COPYRIGHT : Neil D Lawrence, Mehdi Azzouzi (1998, 1999)

% ENSMLP
  
% Check arguments for consistencya
errstring = consist(net, 'mixpars', x, t);
if ~isempty(errstring);
  error(errstring);
end
mixing_coeff = get_pi(net.z);

% Calculate the portion of the entropy due to each component
entropy = 0;
for m = 1:net.M

  entropy = entropy + ...
      mixing_coeff(m)*ensgaussentropy(net.ens(m));
end

% Calculate the portion of the data contribution due to each component
e1 = 0;
for m = 1:net.M
  
  [sexp, qexp] = ensoutputexpec(net.ens(m), x);
  ndata = size(x, 1);
  temp = sum(qexp -2*t.*sexp + t.*t, 1);
  E_D = 0.5*(sum(temp, 2));
  
  
  if isfield(net.ens(1), 'betaposterior')
    ecomp = .5*(sum(net.ens(1).beta.*temp) ...
		- ndata*sumn(net.ens(1).lnbeta, 1) ...
		+ net.ens(1).nout*ndata*log(2*pi));
  else
    ecomp = E_D;
  end
  e1 = e1 + ecomp*mixing_coeff(m);
end

% Evaluate the prior contribution to the error.
e2 = 0;
if isfield(net.ens(m), 'alphaprior')
  
  w = enspak(net.ens(m));
  [expA, explogA] = priorinvcov(net.ens(m));  
  
  switch net.ens(m).covstrct
    
   case 'none'
    TrCA = 0;
    
   case 'diag'     
    diagC = [net.ens(m).d1(:); net.ens(m).db1(:); 
	     net.ens(m).d2(:); net.ens(m).db2(:)].^2;
    TrCA = diagC'*expA;
    
   otherwise
    error('Covariance type not yet implemented')
    
  end 
  
  epriorcomp = 0.5*(w(1:net.ens(m).nwts).^2*expA + TrCA ...
		    + net.ens(m).nwts*log(2*pi)-sum(explogA));
else
  epriorcomp = 0;
end

% Evaluate the mutual information contribution and remove from entropy.
entropy = entropy -  mmi(net);

% disp([e1 e2 entropy])
e = e1 + e2 - entropy;  
 
if e<0
  warning('error is less than zero')
end

