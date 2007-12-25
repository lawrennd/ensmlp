function  [net] = init_R(net)
%INIT_R       initialises the smoothing distributions
%             simply put the values of the R distributions to be the same as
%             the Q ones

tnin = net.nin + 1;
nw1 = net.nhidden*(tnin);
nw2 = (net.nhidden + 1)*net.nout;
nwts = nw1+nw2;

for m = 1:net.M
  net.smooth(m).w1 = net.ens(m).w1;
  net.smooth(m).w2 = net.ens(m).w2;
  net.smooth(m).b1 = net.ens(m).b1;
  net.smooth(m).b2 = net.ens(m).b2;
  switch net.smooth(m).covstrct
   case 'diag'
    %net.smooth(m).d1 = 10000*ones(net.nin + 1, net.nhidden);
    %net.smooth(m).d2 = 10000*ones(net.nhidden + 1, net.nout); 
   case 'symmetric'
    %for n = 1:nwts
    %  s = (n-1)*(nwts+1)-(n-1)*(n)/2+1;
    %  net.smooth(m).U(s) = 10000*net.smooth(m).U(s);
    % end
  end
end

if strcmp(net.soft, 'y')
  mix_coeff = get_pi(net.z);
else
  mix_coeff = net.pi;
end


net.y = log(mix_coeff);