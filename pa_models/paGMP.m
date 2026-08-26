function y = paGMP(x, cfg, profile)
%PAGMP Generalized memory polynomial PA model with lagging envelope terms.
if nargin < 3 || isempty(profile)
    profile = generatePADriftProfile(numel(x), cfg, 'nominal');
end
x = x(:);
if isempty(x), error('Input signal is empty.'); end
if any(~isfinite(real(x))) || any(~isfinite(imag(x)))
    error('Input contains NaN or Inf.');
end
N = numel(x);
y = zeros(N,1);
orders = cfg.pa.orders;
M = cfg.pa.memoryDepth;
L = cfg.pa.gmpLagDepth;
baseCoeffs = cfg.pa.coeffs;

for n = 1:N
    acc = 0;
    commonScale = profile.G(n) * exp(1j*profile.Phi(n));
    for ip = 1:numel(orders)
        p = orders(ip);
        for m = 0:M
            idx = n - m;
            if idx >= 1
                ms = profile.MemoryScale(n,m+1);
                xm = x(idx);
                c = baseCoeffs(ip,m+1) * commonScale * ms * profile.SatScale(n)^(1-p);
                acc = acc + c * xm * abs(xm)^(p-1);
                for ell = 1:L
                    idxLag = idx - ell;
                    if idxLag >= 1 && p > 1
                        crossScale = 0.15 / ell;
                        acc = acc + crossScale*c * xm * abs(x(idxLag))^(p-1);
                    end
                end
            end
        end
    end
    y(n) = acc;
end
if cfg.pa.noiseStd > 0
    y = y + cfg.pa.noiseStd/sqrt(2)*(randn(N,1)+1j*randn(N,1));
end
end
