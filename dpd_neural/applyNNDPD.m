function z = applyNNDPD(x, model, cfg)
%APPLYNNDPD Apply NN-DPD model to desired input x.
x = x(:);
if isempty(x), error('Input signal is empty.'); end
if ~isfield(model,'memoryDepth')
    error('Invalid NN model.');
end
[X, ~, idx] = makeNNFeatures(x, [], cfg);
Xn = (X - model.normalization.mu) ./ model.normalization.sigma;

if isfield(model,'hasDeepLearning') && model.hasDeepLearning && ~isempty(model.net)
    dlX = dlarray(single(Xn), 'CB');
    dlY = forward(model.net, dlX);
    Y = double(gather(extractdata(dlY)));
else
    A = [Xn; ones(1,size(Xn,2))].';
    Y = (A * model.fallbackW).';
end
pred = complex(Y(1,:).', Y(2,:).');
z = x;
z(idx) = pred;
if isfield(model,'outputClip') && ~isempty(model.outputClip)
    z = clipComplexLocal(z, model.outputClip);
end
end

function y = clipComplexLocal(x, limit)
r = abs(x);
scale = min(1, limit ./ max(r, eps));
y = x .* scale;
end
