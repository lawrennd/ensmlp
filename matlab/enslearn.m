function [net, lll, inputs]  = enslearn(nhidden, x, t, alphaprior, betaprior, ...
				rseed, dataset, niters)

% ENSLEARN Learn an ensemble neural network from data.
% FORMAT
% DESC learns an ensemble neural network give inputs and targets. The
% user also provides priors random seeds.
% ARG nhidden : number of hiden units.
% ARG x : input data.
% ARG alphaprior : prior over alphas.
% ARG betaprior : prior over betas.
% ARG rseed : random seed.
% ARG datasetName : dataset name.
% ARG niters : number of iterations.
% RETURN net : learnt model.
% RETURN lll : lower bound on log likelihood.
%
% COPYRIGHT : Neil D. Lawrence, 1999
% 
% SEEALSO : ens, enslll, evidlearn

% ENSMLP
  
% Load data and get parameters for the network
if nargin<8
  niters = 10;
end

nin = size(x, 2);
nout = size(t, 2);
inputs = 1:nin;
randn('seed', rseed)
rand('seed', rseed)

% Normalise input and output


% Generate the matrix of inputs x and targets t.
numTarget = size(t, 1);

% Set up vector of options for the optimiser.
options = zeros(1,18); 			% Default options vector.
options(1) = 1; 			% This provides display of error values
options(2) = 1.0e-4; 			% Absolute precision for weights.
options(3) = 1.0e-4; 			% Precision for objective function.
options(9) = 0;
options(14) = 200; 			% Number of training cycles in
                                        % inner loop. 
%/~
%soft='n';
%~/

net = ens(nin, nhidden, nout, 'linear', 'diag', 0, ...
	  alphaprior, betaprior);
count = 1;
lll(count) = enslll(net, x, t);
%/~
%save([resultsdir 'longer_ens_' dataset '_' num2str(nhidden) 'hidden_' ...
%      alphaprior.type '_iter' num2str(0) '_seed' num2str(rseed) '.mat'], ...
%     'net', 'lll')
%~/

for j = 1:niters
  if j>niters-4
    options(2) = 1e-5;
    options(3) = 1e-5;
  end
  net = netopt(net, options, x, t, 'quasinew');
  count = count + 1;
  lll(count) = enslll(net, x, t);
  fprintf(1,  ['\nlower bound after e-step in w '...
	       'distributions %d:\n'], lll(count));
  net = ensupdatehyperpar(net, x, t);
  net = enshypermoments(net);
  count = count + 1;
  lll(count) = enslll(net, x, t);
  fprintf(1, '  alpha =  %8.5f\n', net.alpha);
  fprintf(1, '  beta  =  %8.5f\n', net.beta);
  fprintf(1,  ['\nlower bound after e-step in hyper alpha and beta '...
	       'distributions %d:\n'], lll(count));
  if options(9)
    hypergradchek(net, x, t)
  end
  if strcmp(alphaprior.type, 'NRD')
    % Mstep in model structure
    % Try removing a hidden or an input node
    newlllIn = -Inf;
    newlllHid = -Inf;
    if net.nin > 1
      alpha_in = net.alpha(1:net.nin);
      [void, inputremovenode] = max(alpha_in);
      netInTry = ensrmnode(net, 1, inputremovenode);
      tempX = x(:, [1:(inputremovenode-1) (inputremovenode+1):net.nin]);
      newlllIn = enslll(netInTry, tempX, t);
    end
    if net.nhidden >1
      alpha_hidden = net.alpha(net.nin+2:net.nin + net.nhidden ...
			       + 1);
      [void, removenode] = max(alpha_hidden);
      netHidTry = ensrmnode(net, 2, removenode);
      newlllHid = enslll(netHidTry, x, t);
    end
    %Calculation is Adjusted for multiple modes
    lllvec = [lll(count)+ log(net.nhidden) + log(2) ...
	      newlllIn + log(net.nhidden) + log(2) ...
	      newlllHid];
    fprintf('Log likelihoods: %f\n', lllvec);
    [void, index] = max(lllvec);
    
    switch index
     case 1
      % do nothing
      count = count + 1;
      lll(count) = lll(count-1);
      
     case 2
      % newlllIn is largest
      fprintf(1, 'pruning input node ....\n');
      x = tempX;
      inputs = inputs(1, [1:(inputremovenode-1) (inputremovenode+1):net.nin]);
      net = netInTry;
      count = count + 1;
      lll(count) = newlllIn;
      
     case 3
      % newlllHid is largest
      fprintf(1, 'pruning hidden node ....\n');
      net = netHidTry;
      count = count + 1;
      lll(count) = newlllHid;
    end
  end
%/~  save([resultsdir 'longer_ens_' dataset '_' num2str(nhidden) 'hidden_' ...
%	alphaprior.type '_iter' num2str(j) '_seed' num2str(rseed) ...
%
%~/'.mat'],'net', 'lll')
end

