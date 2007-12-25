function [net, gamma]  = evidlearn(nhidden, x, t, prior, beta, ...
				   rseed, dataset, niters)
% EVIDLEARN Learn an evidence procedure neural network from data.
% FORMAT
% DESC learns an evidence procedure neural network give inputs and targets. The
% user also provides priors random seeds.
% ARG nhidden : number of hiden units.
% ARG x : input data.
% ARG alphaprior : prior over alphas.
% ARG betaprior : prior over betas.
% ARG rseed : random seed.
% ARG datasetName : dataset name.
% ARG niters : number of iterations.
% RETURN net : learnt model.
%
% COPYRIGHT : Neil D. Lawrence, 1999
% 
% SEEALSO : ens, enslll, enslearn

% ENSMLP

  % Load data and get parameters for the network
if nargin<8
  niters = 10;
end
%/~
%RESULTS = getenv('RESULTS')
%resultsdir = [RESULTS '/matlab/enslab/' dataset '/'];
%~/
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
options(2) = 1.0e-7; 			% Absolute precision for weights.
options(3) = 1.0e-7; 			% Precision for objective function.
options(9) = 0;
options(14) = 500; 			% Number of training cycles in

if isstruct(prior)
  if length(prior.alpha)>4
    priortype = 'ard';
  else
    priortype = 'group';
  end
else
  priortype = 'single';
end



ninner = 10;                     % Number of innter loops.

net = mlp(nin, nhidden, nout, 'linear', prior, beta);
%/~save([resultsdir 'evid_' dataset '_' num2str(nhidden) 'hidden_' ...
%      priortype  '_iter' num2str(0) '_seed' num2str(rseed) '.mat'], 'net')
%~/
for k = 1:niters
  net = netopt(net, options, x, t, 'quasinew');
  [net, gamma(k)] = evidence(net, x, t, ninner);
  fprintf(1, '\nRe-estimation cycle %d:\n', k);
  fprintf(1, '  alpha =  %8.5f\n', net.alpha);
  fprintf(1, '  beta  =  %8.5f\n', net.beta);
  fprintf(1, '  gamma =  %8.5f\n\n', gamma(k));
  disp(' ')
%/~  save([resultsdir 'evid_' dataset '_' num2str(nhidden) 'hidden_' ...
%	priortype '_iter' num2str(k) '_seed' num2str(rseed) '.mat'], 'net', ...
%~/       'gamma')
end







