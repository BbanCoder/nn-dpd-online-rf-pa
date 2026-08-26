function y = paMemorylessSaleh(x, cfg)
%PAMEMORYLESSSALEH Memoryless Saleh PA model.
x = x(:);
if isempty(x), error('Input signal is empty.'); end
if any(~isfinite(real(x))) || any(~isfinite(imag(x)))
    error('Input contains NaN or Inf.');
end
alphaA = 2.0; betaA = 1.0;
alphaP = cfg.pa.ampm; betaP = 0.5;
r = abs(x);
amp = (alphaA*r) ./ (1 + betaA*r.^2);
phi = (alphaP*r.^2) ./ (1 + betaP*r.^2);
y = amp .* exp(1j*(angle(x)+phi));
y = y ./ max(1, rms(y));
end
