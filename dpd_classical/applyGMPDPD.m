function z = applyGMPDPD(x, model)
%APPLYGMPDPD Apply trained GMP-DPD model.
x = x(:);
if isempty(x), error('Input signal is empty.'); end
[Phi, idx] = buildGMPRegressor(x, model.orders, model.memoryDepth, model.lagDepth);
z = x;
z(idx) = Phi * model.coefficients;
if isfield(model,'outputClip') && ~isempty(model.outputClip)
    z = clipComplex(z, model.outputClip);
end
end

function y = clipComplex(x, limit)
r = abs(x);
scale = min(1, limit ./ max(r, eps));
y = x .* scale;
end
