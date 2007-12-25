function [netmix, lll] = mixenslearn(netinit, x, t, ncomp, dataset)

% MIXENSLEARN Learn a mixture nsemble neural network from data.
% FORMAT
% DESC learns an mixture of ensemble neural networks give inputs and targets. The
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
% SEEALSO : mixens, mixenslll, evidlearn

% ENSMLP


% Set up vector of options for the optimiser.
options = zeros(1,18); 			% Default options vector.
options(1) = 1; 			% This provides display of error values.
options(2) = 1.0e-5; 			% Absolute precision for weights.
options(3) = 1.0e-5; 			% Precision for objective function.
options(9) = 0;
 
soft='n';
rseed = 10;
fprintf(1,'Initialise the mixture\n');
% Initialise the mixture
netmix = mixens(ncomp, netinit.nin, netinit.nhidden, netinit.nout, 'linear', 'diag', 'diag', 0, 0,...
                netinit.alphaprior, netinit.betaprior, rseed, soft);

%init each component with the supplied network

for m = 1:ncomp
  netmix.ens(m) = netinit;
  netmix.ens(m).w1 = netmix.ens(m).w1.*(1 + .01*randn(size(netmix.ens(m).w1)));
  netmix.ens(m).b1 = netmix.ens(m).b1.*(1 + .01*randn(size(netmix.ens(m).b1)));
  netmix.ens(m).w2 = netmix.ens(m).w2.*(1 + .01*randn(size(netmix.ens(m).w2)));
  netmix.ens(m).b2 = netmix.ens(m).b2.*(1 + .01*randn(size(netmix.ens(m).b2)));
end

count = 1;
lll(count) = mixenslll(netmix, x, t);

% Init R
fprintf('Init R\n');
netmix = init_R(netmix);
fprintf('MMI = %f\n', mmi(netmix));
fprintf('Now optimize the mmi wrt R\n');
options(14) = 100; 			% Number of training cycles in inner l
netmix.type = 'mmi_R';
netmix = netopt(netmix, options, x, t, 'quasinew');
netmix.type = 'mixens';
count = count + 1;
lll(count) = mixenslll(netmix, x, t);

fprintf('Mutual information after initialisation: %f\n', mmi(netmix));

save(['mixens_' dataset '_' num2str(netinit.nhidden) 'hidden_' num2str(ncomp) 'comps_' netinit.alphaprior.type '_iter' num2str(0) '_seed' num2str(rseed) '.mat'])

% Now optimise combined model
options(14) = 30;
for i = 1:10
  netmix = netopt(netmix, options, x, t, 'quasinew');
  count = count + 1;
  lll(count) = mixenslll(netmix, x, t);
  fprintf('Mutual information after iterating: %f\n', mmi(netmix));
  netmix.type = 'mmi_R';
  netmix = netopt(netmix, options, x, t, 'quasinew');
  fprintf('Mutual information after iterating: %f\n', mmi(netmix));
  count = count + 1; 
  netmix.type = 'mixens';
  lll(count) = mixenslll(netmix, x, t);

  save(['mixens_' dataset '_' num2str(netinit.nhidden) 'hidden_' ...
      num2str(ncomp) 'comps_' netinit.alphaprior.type '_iter' num2str(i) ...
      '_seed' num2str(rseed) '.mat'])
end

