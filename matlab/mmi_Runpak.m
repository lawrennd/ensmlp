function net = mmi_Runpak(net, w)

% MMI_RPAK Place parameters of smoothing distribution in a network.
% FORMAT
% DESC extracts the parameters of the smoothing distributions from the
% network in preparation for optimisation.
% ARG net : the network containing the smoothing distributions.
% ARG w : the parameters of the smoothing distributions.
% RETURN net : the network containing the smoothing distributions.
%
% COPYRIGHT : Mehdi Azzouzi and Neil D. Lawrence, 1998, 1999
%
% SEEALSO : mmi_Rerr, mmi_Rgrad, mmi_Runpak, mixens

% ENSMLP
  
errstring = consist(net, 'mmi_R');
if ~isempty(errstring);
  error(errstring);
end

M = net.M;

% Unpak the mixing and lambda coeff
if strcmp(net.soft, 'y') == 1
  net.z = w(1:M);
  net.y = w(M+1:2*M);
  mark1 = 2*M+1;
else
  net.y = w(1:M);
  mark1 = M+1;
end

% For each component unpak the mixture and the smooth distributions
for m = 1:M
  net.smooth(m) = smoothunpak(net.smooth(m), w(mark1:mark1+net.smooth(m).npars-1));
  mark1 = mark1 + net.smooth(m).npars;
end




