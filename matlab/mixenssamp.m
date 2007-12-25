function [y, sampweights] = mixenssamp(net, x, nsamps)

% MIXENSSAMP Sample from the posterior distribution
% FORMAT
% DESC takes a sample from the posterior distribution of the weights and generates ouputs Y for the specified inputs X
% ARG net : input network.
% ARG x : input locations.
% ARG nsamps : number of samples.
% RETURN y : the output sample locations.
% RETURN sampWeights : the weights used to give the samples.
%
% SEEALSO : ensfwd, ensunpak, enssamp
%
% COPYRIGHT : Neil D. Lawrence, 1998, 1999

% ENSMLP
	
errstring = consist(net, 'mixens', x);
if ~isempty(errstring);
  error(errstring);
end

selectmode = rand(1, nsamps);
selectmode = selectmode*net.M;

for i = 1:net.M
  index = find(selectmode>=(i-1) & selectmode<i);
  nsampsM = length(index);
  [Cuu, Cvv, Cuv] = enscovar(net.ens(i));
  C = [Cuu Cuv; Cuv' Cvv];
  covstrct = net.ens(i).covstrct;
  net.ens(i).covstrct = 'none';
  weights = enspak(net.ens(i));
  sampweights = gsamp(weights, C, nsampsM);
  for j = 1:nsampsM
    sampnet = ensunpak(net.ens(i), sampweights(j, :));
    y(:, index(j)) = ensfwd(sampnet, x);
  end

end
