function y = normalizeSignal(x)
%NORMALIZESIGNAL Normalize complex signal to unit average power.
if nargin < 1 || isempty(x)
    error('Input signal is empty.');
end
x = x(:);
if any(~isfinite(real(x))) || any(~isfinite(imag(x)))
    error('Input signal contains NaN or Inf.');
end
p = mean(abs(x).^2);
if p <= 0
    error('Input signal has zero power.');
end
y = x ./ sqrt(p);
end
