function y = paMemoryPolynomial(x, coeffs, orders, memoryDepth)
%PAMEMORYPOLYNOMIAL Complex baseband memory polynomial PA model.
x = validateComplexVectorLocal(x);
if nargin < 4
    error('Usage: paMemoryPolynomial(x, coeffs, orders, memoryDepth)');
end
N = numel(x);
y = zeros(N,1);
if size(coeffs,1) ~= numel(orders) || size(coeffs,2) ~= memoryDepth+1
    error('Coefficient matrix size must be numel(orders) x (memoryDepth+1).');
end
for n = 1:N
    acc = 0;
    for ip = 1:numel(orders)
        p = orders(ip);
        for m = 0:memoryDepth
            idx = n - m;
            if idx >= 1
                xm = x(idx);
                acc = acc + coeffs(ip,m+1) * xm * abs(xm)^(p-1);
            end
        end
    end
    y(n) = acc;
end
end

function x = validateComplexVectorLocal(x)
if isempty(x), error('Input signal is empty.'); end
x = x(:);
if any(~isfinite(real(x))) || any(~isfinite(imag(x)))
    error('Input contains NaN or Inf.');
end
end
