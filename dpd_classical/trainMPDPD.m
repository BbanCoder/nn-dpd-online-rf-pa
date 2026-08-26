function model = trainMPDPD(paOutput, referenceInput, cfg)
%TRAINMPDPD Train an indirect-learning MP-DPD post-inverse model.
paOutput = paOutput(:); referenceInput = referenceInput(:);
if numel(paOutput) ~= numel(referenceInput)
    error('paOutput and referenceInput must have same length.');
end
[Phi, idx] = buildMPRegressor(paOutput, cfg.dpd.orders, cfg.dpd.memoryDepth);
target = referenceInput(idx);
lambda = cfg.dpd.regularization;
coef = solveRidgeLS(Phi, target, lambda);
model = struct('type','MP','orders',cfg.dpd.orders,'memoryDepth',cfg.dpd.memoryDepth, ...
               'coefficients',coef,'outputClip',cfg.dpd.outputClip);
end
