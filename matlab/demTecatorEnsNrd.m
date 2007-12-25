% DEMTECATORENSARD Demonstrate Ensemble with NRD on Tecator data set.

% ENSMLP

% Train different mixture playing with the number of hidden units and the
% seed
% We use alpha and beta priors corresponding to those used in MCMC
alpha_a = 3e-4;
alpha_b = 1e-3;
beta_a = 0.02;
beta_b = 1e-4;
nseed1 = 1;
nseed2 = 5;
nhidden = 8;
dataset = 'tecator';
load([dataset]);

nin = 10;
nout = 1;

xtrain = [tecatorTrIn(:, 1:nin); tecatorValIn(:, 1:nin)];
ttrain = [tecatorTrOut(:, 1); tecatorValOut(:, 1)];
xtest = tecatorTsIn(:, 1:nin);
ttest = tecatorTsOut(:, 1);

meanxtrain = mean(xtrain);
xtrain = (xtrain - ones(size(xtrain, 1), 1)*meanxtrain);
stdxtrain = std(xtrain);
xtrain = xtrain./(ones(size(xtrain, 1), 1)*stdxtrain);
xtest = (xtest - ones(size(xtest, 1), 1)*meanxtrain) ...
	./(ones(size(xtest, 1), 1)*stdxtrain);
meanttrain = mean(ttrain);
ttrain = ttrain - meanttrain;
stdttrain = std(ttrain);
ttrain = ttrain/stdttrain;
ttest = (ttest - meanttrain)/stdttrain;

clear tecatorTrIn tecatorValIn tecatorTsIn tecatorTrOut ...
    tecatorValOut tecatorTrOut
niters = 20;

alpha_a = sqrt(alpha_a/alpha_b)*sqrt(alpha_b);
alpha_b = sqrt(alpha_b);
alphaprior = enshyperprior(nin, nhidden, nout, 'NRD', ...
			   alpha_a*ones(1, nin), alpha_a, ...
			   alpha_a*ones(1, nhidden), alpha_a, ...
			   alpha_a*ones(1, nout), ...
			   alpha_b*ones(1, nin), alpha_b, ...
			   alpha_b*ones(1, nhidden), alpha_b, ...
			   alpha_b*ones(1, nout));
%/~
%alphaprior = enshyperprior(nin, nhidden, nout, 'ARD', ...
%			   alpha_a*ones(1, nin), alpha_a, alpha_a, ...
%			   alpha_a, [], ...
%			   alpha_b*ones(1, nin), alpha_b, alpha_b, ...
%			   alpha_b, []);

%alphaprior = enshyperprior(nin, nhidden, nout, 'group', ...
%			   alpha_a, alpha_a, alpha_a, alpha_a, [], ...
%			   alpha_b, alpha_b, alpha_b, alpha_b, []);
%~/

betaprior = enshyperprior(nin, nhidden, nout, 'beta', beta_a, beta_b);

% Do it 
for n = nseed1:nseed2
  rseed = n*10;
  [netRes(n), lllLogRes(n, :), inputsRes{n}] = enslearn(nhidden, xtrain, ttrain, ...
                                                    alphaprior, betaprior, ...
                                                    rseed, dataset, niters);
  lllRes(n) = enslll(netRes(n), xtrain(:, inputsRes{n}), ttrain);
  ytest = ensoutputexpec(netRes(n), xtest(:, inputsRes{n}));
  nmsepRes(n) = sum((ttest - ytest).^2)./sum((ttest-mean(ttrain)).^2);
  SEPRes(n) = sqrt(mean((stdttrain*(ttest -ytest)).^2));
end

save([dataset '_ens_' alphaprior.type '_' num2str(nhidden) ...
      'hidden.mat'], 'netRes', 'lllRes', 'nmsepRes', 'inputsRes', 'SEPRes')




