function prior = enshyperprior(nin, nhidden, nout, type, ...
                               aI, ab1, aH, ab2, aO, ...
                               bI, bb1, bH, bb2, bO)

% ENSHYPERPRIOR Create gamma priors for hyperparameters in ensmeble learning.
% FORMAT
% DESC Create gamma priors for hyperparameters in ensmeble learning.
% ARG nin : number of inputs.
% ARG nhidden : number of hidden nodes.
% ARG nout : number of outputs.
% ARG type : should be one of 'beta', 'single', 'group', 'ARD' or 'NRD'.
% ARG ai : 
% ARG ab1 : 
% ARG ah :
% ARG ab2 :
% ARG a0 :
% ARG bi :
% ARG bb1 :
% ARG bh :
% ARG bb2 :
% ARG b0 :
% ARG PRIOR = ENSPRIOR(NIN, NHIDDEN, NOUT, TYPE, ...
%       AI, AB1, AH, AB2, AO, BI, BB1, BH, BB2, BO) 
% RETURN prior : a data structure PRIOR, with fields PRIOR.A, PRIOR.B and 
% 	PRIOR.TYPE. 
% 
% SEEALSO : ens, enserr, ensgrad, evidenc
% 
% BASEDON : Christopher M. Bishop, 1996
%
% BASEDON : Ian T. Nabney, 1996, 1997
%
% COPYRIGHT : Neil D. Lawrence, 1999

% ENSMLP

types = {'beta', 'single', 'group', 'ARD', 'NRD'};

if sum(strcmp(type, types)) == 0
  error('Undefined prior type. Exiting.');
else
  prior.type = type;
end

nwts = nin*nhidden + nhidden + (nhidden + 1)*nout;
mark1 = nin*nhidden;

switch prior.type

case 'beta' % Create prior for the inverse noise variance
  
  if nargin ~=6 
    error('Incorrect number of arguments')
  end
  if size(aI)~=[1, nout] 
    error('Parameter a dimensions not compatible with number of outputs')
  end
  if size(ab1)~=[1, nout]
    error('Parameter b dimensions not compatible with number of outputs')
  end
  prior.a = aI;
  prior.b = ab1;
  
  
case 'single' % Prior over alpha involves just a single parameter.
  
  if nargin ~=6 
    error('Incorrect number of arguments')
  end
  if size(aI)~=[1, 1] 
    error('Parameter a dimensions not compatible with prior type.')
  end
  if size(ab1)~=[1, 1]
    error('Parameter b dimensions not compatible with prior type.')
  end
   
  prior.a = aI;
  prior.b = ab1;


case {'group', 'ARD', 'NRD'}
  
  switch prior.type
   case {'group'}
    if size(aI) ~= [1,1] 
      error('Parameter aI dimensions not compatible with prior type.')
    end
    indx = [ones(mark1, 1); zeros(nwts-mark1, 1)];
    
   case {'ARD', 'NRD'}
    if size(aI) ~= [1, nin]
      error('Parameter aI dimensions not compatible with prior type.')
    end
    
    indx = [kron(ones(nhidden, 1), eye(nin)); zeros(nwts-mark1, nin)];
  end
  
  mark2 = mark1 + nhidden;
  indx = [indx [zeros(mark1, 1); ones(nhidden, 1); zeros(nwts-mark2, 1)]];
  mark3 = mark2 + nhidden*nout;
  
  switch prior.type
   case {'group', 'ARD'}    
    if size(aH) ~= [1,1] 
      error('Parameter aH dimensions not compatible with prior type.')
    end
    
    indx = [indx [zeros(mark2, 1); ones(nhidden*nout, 1); ...
		  zeros(nwts-mark3, 1)]];
    
   case 'NRD'  
    if size(aH) ~= [1, nhidden]
      error('Parameter aH dimensions not compatible with prior type.')
    end
    
    block = zeros(nhidden*nin, nhidden);
    
    for i = 1:nhidden
      block((i-1)*nin + (1:nin), i) = ones(nin, 1);
    end
    
    block = [block; kron(ones(nout+1, 1), eye(nhidden)); zeros(nout, nhidden)];
    indx = [indx block];
    
  end
  
  mark4 = mark3 + nout;
  indx = [indx [zeros(mark3, 1); ones(nout, 1)]];
  switch prior.type
    
   case {'group', 'ARD'}
    if ~isempty(aO) 
      error('No prior for the outputs is required.')
    end
    prior.index = sparse(indx);
    prior.a = [aI, ab1, aH, ab2]';
    prior.b = [bI, bb1, bH, bb2]';
    
   case 'NRD'
    if size(aO) ~= [1, nout]
      error('Parameter aO dimensions not compatible with prior type.')
    end
    
    block = zeros(nout*nhidden, nout);
    for i = 1:nout
      block((i-1)*nhidden + (1:nhidden), i) = ones(nhidden, 1);
    end
    block = [zeros((nin+1)*nhidden, nout); block; eye(nout)];
    indx = [indx block];
    prior.index = sparse(indx);
    prior.a = [aI, ab1, aH, ab2, aO]';
    prior.b = [bI, bb1, bH, bb2, bO]';

  end
  


end







