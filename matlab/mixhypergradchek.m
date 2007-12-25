function mixhypergradchek(net, x, t)

% MIXHYPERGRADCHEK Check gradient of hyper parameters.
  
% ENSMLP
  
epsilon = 1.0e-6;
%net = ensupdatehyperpar(net, x, t);
%net = enshypermoments(net);
w = mixenspakpar(net);
nparams = length(w);
deltaf = zeros(1, nparams);
step = zeros(1, nparams);
for i = 1:length(w)
  % Move a small way in the ith coordinate of w
  wold = w;
  w(i) = w(i) + epsilon;
  net = mixensunpakpar(w, net);
  fplus = mixenslll(net, x, t);
  w(i) = w(i) - 2*epsilon;
  net = mixensunpakpar(w, net);
  fminus = mixenslll(net, x, t);
  % Use central difference formula for approximation
  deltaf(i) = 0.5*(fplus - fminus)/epsilon;
  w = wold;
end
fprintf(1, 'Checking gradient ...\n\n');
fprintf(1, '   diffs    \n');
disp([deltaf'])

function w = mixenspakpar(net)

w = [net.ens(1).alphaposterior.a; net.ens(1).betaposterior.a; ...
     net.ens(1).alphaposterior.b; net.ens(1).betaposterior.b];

function net = mixensunpakpar(w, net);
for m = 1:net.M
  mark1 = length(net.ens(1).alpha);
  net.ens(m).alphaposterior.a = w(1:mark1);
  mark2 = mark1 + length(net.ens(1).beta);
  net.ens(m).betaposterior.a = w(mark1+1:mark2);
  mark3 = mark2 + length(net.ens(1).alpha);
  net.ens(m).alphaposterior.b = w(mark2+1:mark3);
  mark4 = mark3 + length(net.ens(1).beta);
  net.ens(m).betaposterior.b = w(mark3+1:mark4);
end
net = mixenshypermoments(net);






