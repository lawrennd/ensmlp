function [sexp, qexp] = mixensoutputexpec(net, x)

% MIXENSOUTPUTEXPEC for each component gives the expectation of the output.
% FORMAT
% DESC gives the expectation of the output function associated with each
% component and its square.
% ARG net : the network from which the expectations are taken.
% ARG x : the input locations for which the outputs are desired.
% RETURN y : the expected output associated with the given input.
% RETURN y2exp : the expectation of the square of the ouptut.
%
% COPYRIGHT : Neil D. Lawrence, 1998, 1999
%
% SEEALSO : ensoutputexpec

% ENSMLP

qexp = zeros(size(x, 1), net.ens(1).nout);
sexp = zeros(size(x, 1), net.ens(1).nout);

% Check the type of softmax
if strcmp(net.soft, 'y') == 1
  mixing_coeff = get_pi(net.z);
else
  mixing_coeff = net.pi;
end
  
for m = 1:net.M
  [compsexp, compqexp] = ensoutputexpec(net.ens(m), x);
  sexp = sexp + mixing_coeff(m)*compsexp;
  qexp = qexp + mixing_coeff(m)*compqexp;
end
