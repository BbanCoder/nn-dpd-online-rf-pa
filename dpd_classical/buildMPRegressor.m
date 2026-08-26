function [Phi, validIdx] = buildMPRegressor(x, orders, memoryDepth)
%BUILDMPREGRESSOR Build memory polynomial regressor matrix.
x = x(:);
if isempty(x), error('Input signal is empty.'); end
if any(~isfinite(real(x))) || any(~isfinite(imag(x)))
    error('Input contains NaN or Inf.');
end
N = numel(x);
validIdx = (memoryDepth+1:N).';
Phi = zeros(numel(validIdx), numel(orders)*(memoryDepth+1));
col = 1;
for p = orders(:).'
    for m = 0:memoryDepth
        xm = x(validIdx-m);
        Phi(:,col) = xm .* abs(xm).^(p-1);
        col = col + 1;
    end
end
end
