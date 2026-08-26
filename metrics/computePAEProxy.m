function pae = computePAEProxy(y, x)
%COMPUTEPAEPROXY Approximate efficiency proxy based on RF output and input power.
y = y(:); x = x(:);
N = min(numel(y), numel(x));
Pout = mean(abs(y(1:N)).^2);
Pin = mean(abs(x(1:N)).^2);
Pdc = 1 + Pout;
pae = max(Pout - Pin, 0) / max(Pdc, eps);
end
