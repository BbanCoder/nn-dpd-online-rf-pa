function [Phi, validIdx] = buildGMPRegressor(x, orders, memoryDepth, lagDepth)
%BUILDGMPREGRESSOR Build a simplified GMP regressor with lagging envelope terms.
x = x(:);
if isempty(x), error('Input signal is empty.'); end
if any(~isfinite(real(x))) || any(~isfinite(imag(x)))
    error('Input contains NaN or Inf.');
end
N = numel(x);
maxDelay = memoryDepth + lagDepth;
validIdx = (maxDelay+1:N).';
nBase = numel(orders)*(memoryDepth+1);
nCross = sum(orders > 1) * (memoryDepth+1) * lagDepth;
Phi = zeros(numel(validIdx), nBase+nCross);
col = 1;
for p = orders(:).'
    for m = 0:memoryDepth
        xm = x(validIdx-m);
        Phi(:,col) = xm .* abs(xm).^(p-1);
        col = col + 1;
    end
end
for p = orders(:).'
    if p <= 1, continue; end
    for m = 0:memoryDepth
        xm = x(validIdx-m);
        for ell = 1:lagDepth
            xe = x(validIdx-m-ell);
            Phi(:,col) = xm .* abs(xe).^(p-1);
            col = col + 1;
        end
    end
end
end
