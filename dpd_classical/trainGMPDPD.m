function model = trainGMPDPD(paOutput, referenceInput, cfg)
%TRAINGMPDPD Train an indirect-learning GMP-DPD post-inverse model.
paOutput = paOutput(:); referenceInput = referenceInput(:);
if numel(paOutput) ~= numel(referenceInput)
    error('paOutput and referenceInput must have same length.');
end
[Phi, idx] = buildGMPRegressor(paOutput, cfg.dpd.orders, cfg.dpd.memoryDepth, cfg.dpd.gmpLagDepth);
target = referenceInput(idx);
lambda = cfg.dpd.regularization;
coef = solveRidgeLS(Phi, target, lambda);
model = struct('type','GMP','orders',cfg.dpd.orders,'memoryDepth',cfg.dpd.memoryDepth, ...
               'lagDepth',cfg.dpd.gmpLagDepth,'coefficients',coef,'outputClip',cfg.dpd.outputClip);
end
