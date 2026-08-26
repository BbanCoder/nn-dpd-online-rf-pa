function y = paDynamicModel(x, cfg, profile, modelType)
%PADYNAMICMODEL Dynamic complex baseband PA model with time-varying MP/GMP coefficients.
if nargin < 4 || isempty(modelType)
    modelType = cfg.pa.model;
end
if nargin < 3 || isempty(profile)
    profile = generatePADriftProfile(numel(x), cfg, 'nominal');
end
x = x(:);
if isempty(x), error('Input signal is empty.'); end
if any(~isfinite(real(x))) || any(~isfinite(imag(x)))
    error('Input contains NaN or Inf.');
end
if numel(profile.G) ~= numel(x)
    error('Profile length must match signal length.');
end

switch upper(modelType)
    case 'GMP'
        y = paGMP(x, cfg, profile);
        return;
    case 'RAPP'
        y0 = paMemorylessRapp(x, cfg);
        y = profile.G(:) .* exp(1j*profile.Phi(:)) .* y0;
        return;
end

N = numel(x);
y = zeros(N,1);
orders = cfg.pa.orders;
M = cfg.pa.memoryDepth;
baseCoeffs = cfg.pa.coeffs;

for n = 1:N
    acc = 0;
    commonScale = profile.G(n) * exp(1j*profile.Phi(n));
    for ip = 1:numel(orders)
        p = orders(ip);
        for m = 0:M
            idx = n - m;
            if idx >= 1
                xm = x(idx);
                memScale = profile.MemoryScale(n,m+1);
                satScale = max(profile.SatScale(n), 0.2);
                c = baseCoeffs(ip,m+1) * commonScale * memScale * satScale^(1-p);
                acc = acc + c * xm * abs(xm)^(p-1);
            end
        end
    end
    y(n) = acc;
end
if cfg.pa.noiseStd > 0
    y = y + cfg.pa.noiseStd/sqrt(2)*(randn(N,1)+1j*randn(N,1));
end
end
