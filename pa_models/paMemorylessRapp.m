function y = paMemorylessRapp(x, cfg)
%PAMEMORYLESSRAPP Memoryless Rapp PA model with simple AM/PM distortion.
if nargin < 2 || isempty(cfg)
    error('Configuration is required.');
end
x = validateComplexVectorLocal(x);
A = cfg.pa.satLevel;
p = cfg.pa.rappSmoothness;
g = cfg.pa.smallSignalGain;
r = abs(g*x);
amp = r ./ (1 + (r./A).^(2*p)).^(1/(2*p));
phase = angle(x) + cfg.pa.ampm * (r.^2 ./ (A^2 + r.^2));
y = amp .* exp(1j*phase);
end

function x = validateComplexVectorLocal(x)
if isempty(x), error('Input signal is empty.'); end
x = x(:);
if any(~isfinite(real(x))) || any(~isfinite(imag(x)))
    error('Input contains NaN or Inf.');
end
end
