% DEMTECATORGPRBFARD Demonstrate GP with RBF ARD prior on Tecator data set.

% ENSMLP

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

clear tecatorTrIn tecatorValIn tecatorTsIn tecatorTrOut tecatorValOut tecatorTrOut

options = gpOptions('ftc');
options.kern = {'rbfard', 'bias', 'white'};
model = gpCreate(nin, 1, xtrain, ttrain, options);
model = gpOptimise(model, 2, 1000);
lllRes= gpLogLikelihood(model);
ytest = gpPosteriorMeanVar(model, xtest);
nmsepRes = sum((ttest - ytest).^2)./sum((ttest-mean(ttrain)).^2);
SEPRes = sqrt(mean((stdttrain*(ttest -ytest)).^2));
