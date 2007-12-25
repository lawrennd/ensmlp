function [y, z, a] = ensfwd(net, x)

% ENSFWD Forward propagation through 2-layer network.
% FORMAT
% DESC takes a network data structure NET together with a
% matrix X of input vectors, and forward propagates the inputs through
% the network to generate a matrix Y of output vectors. Each row of X
% corresponds to one input vector and each row of Y corresponds to one
% output vector.
% ARG net : the network for which the output is required.
% ARG x : the input location for which the output is required.
% RETURN y : the output prediction.
%
% DESC also generates a matrix Z of the hidden unit
% ARG net : the network for which the output is required.
% ARG x : the input location for which the output is required.
% RETURN y : the output prediction.
% RETURN z : the hidden activations.
%
% DESC also returns a matrix A giving the summed inputs to each output
% ARG net : the network for which the output is required.
% ARG x : the input location for which the output is required.
% RETURN y : the output prediction.
% RETURN z : the hidden activations.
% RETURN a : inputs to each output.
% 
% SEEALSO : ens, enspak, ensunpak, enserr, ensbkp, ensgrad
%
% BASEDON : Christopher M Bishop and Ian T Nabney, 1996, 1997
%
% COPYRIGHT : Neil D. Lawrence, 1998

% ENSMLP

% Check arguments for consistency
errstring = consist(net, 'ens', x);
if ~isempty(errstring);
  error(errstring);
end
fact = sqrt(2)/2;
ndata = size(x, 1);

z = erf(fact*(x*net.w1 + ones(ndata, 1)*net.b1));
a = z*net.w2 + ones(ndata, 1)*net.b2;

switch net.actfn

  case 'linear'        %Linear outputs

    y = a;

  case 'logistic'    % Logistic outputs

    y = 1./(1 + exp(-a));

  case 'softmax'   % Softmax outputs
  
    temp = exp(a);
    nout = size(a,2);
    y = temp./(sum(temp,2)*ones(1,nout));

end

