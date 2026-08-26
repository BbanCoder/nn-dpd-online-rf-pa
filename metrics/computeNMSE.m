function nmseDb = computeNMSE(y, ref)
%COMPUTENMSE Normalized mean square error in dB after complex gain alignment.
y = y(:); ref = ref(:);
if isempty(y) || isempty(ref), error('Signals must be non-empty.'); end
N = min(numel(y), numel(ref));
y = y(1:N); ref = ref(1:N);
if any(~isfinite(real(y))) || any(~isfinite(imag(y))) || any(~isfinite(real(ref))) || any(~isfinite(imag(ref)))
    error('Signals contain NaN or Inf.');
end
g = (y' * ref) / max(y' * y, eps);
yA = g * y;
nmse = sum(abs(ref - yA).^2) / max(sum(abs(ref).^2), eps);
nmseDb = 10*log10(max(nmse, realmin));
end
