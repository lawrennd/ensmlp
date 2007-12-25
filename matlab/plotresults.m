printyes=0
HOME = getenv('HOME')
RESULTS = getenv('RESULTS')
resultsdir = ([RESULTS '/matlab/enslab/bayesian_comparison/'])
load([resultsdir 'messyresults3.mat'])
resultsdir = ([RESULTS '/matlab/mixenslab/bayesian_comparison/'])
load([resultsdir 'mixens_bayesian_comparison_5hidden_5comps_group_iter10_seed10.mat'])
span = [-.5 1.5];
plotvals = linspace(span(1), span(2), 100)';
[yout, yout2] = mixensoutputexpec(netmix, plotvals);

hmcsamples = samples;
sigmamix = sqrt(yout2 -  yout.^2);
% Plot comittee positions and ens position.
figure
plot(plotvals, ensoutputexpec(netens, plotvals), 'k-', 'Linewidth', 2)
hold on
grid on
%plot(plotvals, yout+sigmamix, 'k--', 'Linewidth', 2)
%plot(plotvals, yout-sigmamix, 'k--', 'Linewidth', 2)
plot(x, t, 'bo')

for i = 1:netmix.M
  [youtens] = ensoutputexpec(netmix.ens(i), plotvals);
  plot(plotvals, youtens, 'r--', 'Linewidth', 2)
end
axis([span -.5 1.5])
title('(a)')
if printyes 
  printlatex([HOME '\tex\projects\diagrams\noiseysine_commiteepos.eepic'], 6, 'bottom', ...
	   'y')
end
% Plot samples from the mixture distribution and mean and variance
figure
plot(x, t, 'bo')
hold on
grid on
somesamps = mixenssamp(netmix, plotvals, 40);
plot(plotvals, somesamps, 'r:')
plot(plotvals, yout, 'k-', 'Linewidth', 2)
plot(plotvals, yout+sigmamix, 'k--', 'Linewidth', 2)
plot(plotvals, yout-sigmamix, 'k--', 'Linewidth', 2)
axis([span -.5 1.5])
title('(b)')
if printyes 
  printlatex([HOME '\tex\projects\diagrams\noiseysine_commiteesamps.eepic'], 6, 'bottom', ...
	     'y')
end
% Plot samples from the laplace approximation with true mean
figure
sigmlp = zeros(length(plotvals), 1);
g = ensderiv(net, plotvals)/4;
for n = 1 : length(plotvals)
  grad = g(n, :);
  sigmlp(n) = grad*invhess*grad';
end
sigmlp = sqrt(ones(size(sigmlp))./net.beta + sigmlp);

ymlp = ensoutputexpec(net, plotvals);
clear yexp exppart
nsamps = 40
w = enspak(net);
Sigmauu = invhess(1:10, 1:10);
Sigmauv = invhess(1:10, 11:16);
Sigmavv = invhess(11:16, 11:16);
Cuu = hess(1:10, 1:10);
Cuv = hess(1:10, 11:16);
Cvv = hess(11:16, 11:16);
wsamp = gsamp(w, invhess, nsamps);
%[Sigmauu, zeros(10, 6); zeros(6, 10) Sigmavv], nsamps);
clear ysamp
for i = 1:nsamps
  sampnet = ensunpak(net, wsamp(i, :));
  %sampnet.b2 = net.b2;
  %sampnet.w2 = net.w2;
  sampnet.covstrct = 'none';
  sampnet.type = 'ens';
  ysamp(:, i) = ensoutputexpec(sampnet, plotvals);
end
% compute expected value under laplace approximation
Sigmauu = invhess(1:10, 1:10);
Sigmauv = invhess(1:10, 11:16);
Sigmavv = invhess(11:16, 11:16);

vbar = [net.w2; net.b2];
ubar = [net.w1'; net.b1'];
xTu = plotvals*net.w1 + nrepmat(net.b1, 1, size(plotvals, 1));
nw1 = net.nin*net.nhidden;

fact = 1/sqrt(2);
clear g
mark2 = 0;
for row =1:net.nhidden
  mark1 = mark2;
  mark2 = mark1 + net.nin;
  rowrange = (mark1+1):mark2;
  biasrow = nw1 + row;
  xTSigmauv = plotvals*Sigmauv(rowrange, row) + Sigmauv(biasrow, row);
  aTa = sum(plotvals.*(plotvals*Sigmauu(rowrange, rowrange)), 2) + ...
      2*plotvals*Sigmauu(rowrange, biasrow) + ...
      nrepmat(Sigmauu(biasrow, biasrow), 1, size(plotvals, ...
						  1));
  
  a0 = xTu(:, row);
  % the expected value of the hidden node for each data point
  oneplus = 1+aTa;
  sqrtOneplus = sqrt(oneplus);
  g(:, row) = erf(fact*a0./sqrtOneplus);
  exppart(:, row) = sqrt(2/pi)*xTSigmauv./sqrtOneplus...
	.* exp(-.5*a0.*a0./oneplus);
end
for p = 1:net.nout  
  yexp(:, p) = g * net.w2(:, p) + nrepmat(net.b2(1, p), 1, size(plotvals, ...
						  1)) + sum(exppart, 2);
end

plot(x, t, 'ok')
hold on;
grid on
plot(plotvals, ysamp, 'r:', 'LineWidth', 1);
plot(plotvals, yexp, 'k-', 'LineWidth', 2);
%plot(plotvals, ymlp, 'b-', 'LineWidth', 2)
%plot(plotvals, ymlp + sigmlp, 'k--', 'LineWidth', 2);
%plot(plotvals, ymlp - sigmlp, 'k--', 'LineWidth', 2);
axis([span -.5 1.5])
title('(c)')
if printyes 
  printlatex([HOME '\tex\projects\diagrams\noiseysine_laplacemeansamps.eepic'], 6, 'bottom', ...
	     'y')
end


% Now print ens samples and approximation
figure
ysamp = enssamp(netens, plotvals, 40);
[yens yens2] = ensoutputexpec(netens, plotvals);
sigmaens = sqrt(yens2 - yens.^2);
plot(x, t, 'ok')
hold on;
grid on
plot(plotvals, ysamp, 'r:', 'LineWidth', 1);
plot(plotvals, yens, 'k-', 'LineWidth', 2);
plot(plotvals, yens+sigmaens, 'k--', 'Linewidth', 2)
plot(plotvals, yens-sigmaens, 'k--', 'Linewidth', 2)
axis([span -.5 1.5])
title('(d)')
if printyes 
  printlatex([HOME '\tex\projects\diagrams\noiseysine_enssamps.eepic'], 6, 'bottom', ...
	     'y')
end


% Now plot HMC samples
figure
startsample = 10;
lastsample = 50;
numsamples = lastsample - startsample;
varhmc = zeros(size(plotvals));
hmcysamp = zeros(length(plotvals), numsamples);
for i = startsample:lastsample
  k = i - startsample + 1;
  w2 = hmcsamples(k,:);
  nethmc2 = ensunpak(nethmc, w2);
  hmcysamp(:, k) = ensoutputexpec(nethmc2, plotvals);
end
meanhmc = mean(hmcysamp, 2);
plot(x, t, 'ok')
hold on;
grid on
plot(plotvals, hmcysamp, 'r:','linewidth', 1)
plot(plotvals, meanhmc, 'k-', 'linewidth', 2)
axis([span -.5 1.5])
title('(e)')
if printyes 
  printlatex([HOME '\tex\projects\diagrams\noiseysine_hmcsamps.eepic'], 6, 'bottom', ...
	     'y')
end

% Plot along directions between mixture components
clear errmg errsg errenst errlsg
tempnet = netmix.ens(1);
tempnet.npars = tempnet.nwts;
tempnet.covstrct = 'none';
stepsize = 0.005;
indexspan = -200:200;
for i = 1:netmix.M
  [Cuu, Cvv]  = enscovar(netmix.ens(i)); 
  mixenscov{i} = [diag(Cuu); diag(Cvv)];
  tempnet2 = netmix.ens(i);
  tempnet2.covstct ='none';
  tempnet2.npars =16;
  mixensmean{i} = enspak(tempnet2);
  mixensmean{i} = mixensmean{i}(1:16);
end
j=0;
meanmlp = enspak(net);
covmlp = invhess;
ensmean = enspak(tempnet2);
ensmean = ensmean(1:16);
[Cuu, Cvv]  = enscovar(netens); 
enscov = [diag(Cuu); diag(Cvv)];

for m = 1:netmix.M
  for n = (m+1):netmix.M
    j = j+1;
    tempnet2 = netmix.ens(m);
    tempnet2.covstrct ='none';
    tempnet2.npars = 16;
    tempnet3 = netmix.ens(n);
    tempnet3.covstrct ='none';
    tempnet3.npars = 16;
    
    direction = enspak(tempnet2) - enspak(tempnet3);
    tempmean = enspak(tempnet2) - direction/2;
    dist(j) = sqrt(sum(direction.^2));
    direction = direction/dist(j);
    dist(j) = 1;
    for i = indexspan
      plotw = tempmean+i*stepsize*direction*dist(j);
      tempnet = ensunpak(tempnet, plotw);
      errenst(j, i+max(indexspan)+1) = -enserr(tempnet, x, t);
      [void, errsg(j, i+max(indexspan)+1)] = gaussian(ensmean, enscov', plotw);
      [void, errlsg(j, i+max(indexspan)+1)] = gaussian(meanmlp, covmlp, plotw);      errmg(j, i+max(indexspan)+1) = 0;
      for p = 1:netmix.M
	[void, temp(p)] = gaussian(mixensmean{p}, mixenscov{p}', plotw);
	temp(p) = temp(p) + log(netmix.pi(p));
      end
	errmg(j, i+max(indexspan)+1) = log(sum(exp(temp)));
    end
  end
end
figure
i=0;
for m = 1:netmix.M
  for n = (m+1):netmix.M
    i = i+ 1;
    subplot(netmix.M-1, netmix.M-1, (m-1)*(netmix.M-1)+(n-1))
    plot(indexspan*stepsize*dist(i), errmg(i, :) - max(errmg(i, :)), 'r:', 'linewidth', .5)
    hold on
    plot(indexspan*stepsize*dist(i), errenst(i, :) - max(errenst(i, :)), 'b-', 'linewidth', .5)
    %plot(indexspan*stepsize*dist(i), errsg(i, :)- max(errmg(i, :)), 'r:', 'linewidth', 1)
    axis([min(indexspan)*stepsize, max(indexspan)*stepsize, -4, 0.5])  
    YTick = get(gca, 'YTick');
    XTick = get(gca, 'XTick');
    if m == 1
      title(num2str(n))
    end
    if n == m+1
      % do nothing
    else

      set(gca, 'XTickLabel', 32*ones(size(XTick)))
      set(gca, 'YTickLabel', 32*ones(size(YTick)))    
    end
  end
end
subplot(4, 4, 1)
ylabel('1')
subplot(4, 4, 5)
ylabel('2')
subplot(4, 4, 9)
ylabel('3')
subplot(4, 4, 13)
ylabel('4')

if printyes
  printlatex([HOME '\tex\projects\diagrams\mixapprox_transcomps.eepic'], 12, 'top')
end


% Now plot the Laplace aproximation across the eigen values
clear errt errlsg
[v, d] = eig(invhess);
[void, order] = sort(diag(d))
order = order(16:-1:1)
stepsize = 0.02;
indexspan = -70:70;
w = enspak(net);
for j = 1:length(w)
  for i = indexspan
    plotw = w+i*stepsize*v(:, j)';
    tempnet = ensunpak(net, plotw);
    errt(j, i+max(indexspan)+1) = -enserr(tempnet, x, t);
    [void, errlsg(j, i+max(indexspan)+1)] = gaussian(w, invhess, plotw);
  end
end
figure
for i = 1:16
  
  subplot(4, 4, i)
  number = order(i);
  %  title(['Eigenvalue number ' num2str(i)])
plot(indexspan*stepsize, errlsg(number, :)  - max(errlsg(number, :)), 'r:', 'linewidth', .5)
  hold on
  plot(indexspan*stepsize, errt(number, :) - max(errt(number, :)), 'b-', 'linewidth', .5)
  axis([min(indexspan)*stepsize, max(indexspan)*stepsize, ...
	-4, 0.5])  
  YTick = get(gca, 'YTick');
  XTick = get(gca, 'XTick');
  if i < 13
    if mod(i, 4) == 1 
      set(gca, 'XTickLabel', 32*ones(size(XTick)))
      % do nothing
    else
      set(gca, 'XTickLabel', 32*ones(size(XTick)))
      set(gca, 'YTickLabel', 32*ones(size(YTick)))    
    end
   end 
  if i > 12
   % do nothing
    if mod(i, 4) == 1 
      % do nothing
    else
      set(gca, 'YTickLabel', 32*ones(size(YTick)))    
   
    end
  end  
end

if printyes
  printlatex([HOME '\tex\projects\diagrams\laplaceapprox_eigvals.eepic'], 12, 'top')
end

