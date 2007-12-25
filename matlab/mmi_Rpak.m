function p = mmi_Rpak(net)

% MMI_RPAK Extract parameters of smoothing distributions from network.
% FORMAT
% DESC extracts the parameters of the smoothing distributions from the
% network in preparation for optimisation.
% ARG net : the network containing the smoothing distributions.
% RETURN w : the parameters of the smoothing distributions.
%
% COPYRIGHT : Mehdi Azzouzi and Neil D. Lawrence, 1998, 1999
%
% SEEALSO : mmi_Rerr, mmi_Rgrad, mmi_Runpak, mixens

% ENSMLP

% Check arguments for consistency

errstring = consist(net, 'mmi_R');
if ~isempty(errstring);
  error(errstring);
end

M = net.M;

% Pak the mixing and lambda coefficients
if strcmp(net.soft, 'y') == 1
  p = [net.z net.y];
else
  p = [net.y];
end

% For each component pak the mixture and the smooth distributions
for m = 1:M
  p = [p smoothpak(net.smooth(m))];
end




