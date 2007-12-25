function net = smooth(nin, nhidden, nout, strct, t)
% SMOOTH Create the smoothing distribtuions for the mutual information bound.
% FORMAT
% DESC create the smoothing distributions for the bound on the mutual
% information.
% ARG nin : number of input dimensions.
% ARG nhidden : number of hidden units.
% ARG nout : number of output dimensions.
% ARG covStruct : covariance structure of distributions.
% ARG t : output data.
%
% COPYRIGHT : Neil D. Lawrence, 1998, 1999
%
% SEEALSO : smoothunpak, smoothpak, smootherr, smoothgrad

% ENSMLP

net.type = 'smooth';
net.nin = nin;
net.nhidden = nhidden;
net.nout = nout;
net.nwts = (nin + 1)*nhidden + (nhidden + 1)*nout;
covstrcts = {'full', 'layered', 'noded', 'unnoded', 'diag', 'symmetric'};

% allow big values for the covariance elements
amplitude = 1;

% Check the covariance structure
if sum(strcmp(strct, covstrcts)) == 0
  error('Undefined covariance structure. Exiting.');
else 
  net.covstrct = strct;
  switch  strct
    % Diagonal matrix
    case 'diag'
      net.d1 = amplitude*rand(nin + 1, nhidden);
      net.d2 = amplitude*rand(nhidden + 1, nout); 
      net.npars = net.nwts + (nin + 1)*nhidden + ...
	  (nhidden+ 1)*nout; 

    % Special form matrix R = diag(u) - \sum mu.mu'
    case 'unnoded'
      net.d1 = amplitude*rand(nin + 1, nhidden);
      net.d2 = amplitude*rand(nhidden + 1, nout); 
      net.t = t;
      net.mu1 = amplitude*randn(nin + 1, nhidden, t);
      net.mu2 = amplitude*randn(nhidden + 1, nout, t);
      net.npars = net.nwts + (nin + 1)*nhidden*(t + 1) + ...
	  (nhidden+ 1)*nout*(t + 1); 
    
    % Symmetric matrix R = U.U'
    case 'symmetric'
      net.U = amplitude*randn(net.nwts*(net.nwts+1)/2,1);
      net.npars = net.nwts + net.nwts*(net.nwts+1)/2;
     
    % Full matrix
    case 'full'
      net.d1 = amplitude*randn((nin + 1)*nhidden, (nin + 1)*nhidden);
      net.d2 = amplitude*randn((nhidden + 1)*nout, (nhidden + 1)*nout);
      net.npars = net.nwts + ((nin + 1)*nhidden)^2 + ...
	  ((nhidden+ 1)*nout)^2; 
    otherwise
      error('Covariance structure not yet implemented.\n');
  end
end

% Initialise the mean of the weights and biases
net.w1 = randn(nin, nhidden)/sqrt(nin + 1);
net.b1 = randn(1, nhidden)/sqrt(nin + 1);
net.w2 = randn(nhidden, nout)/sqrt(nhidden + 1);
net.b2 = randn(1, nout)/sqrt(nhidden + 1);


