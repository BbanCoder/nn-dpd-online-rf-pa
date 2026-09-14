function [doAdapt, action] = adaptationScheduler(metrics, cfg, state)
%ADAPTATIONSCHEDULER Decide whether to adapt, based on a relative NMSE
%   degradation trigger plus cooldown and budget constraints.
%   The trigger fires when the block NMSE is more than
%   cfg.adaptation.relNMSE_dB worse than the best NMSE observed so far
%   (state.bestNMSE). Absolute thresholds (cfg.thresholds) are conformity
%   criteria reported in the results; using them as triggers would fire on
%   every block at this drive level and waste the adaptation budget on
%   healthy blocks.
if nargin < 3 || isempty(state)
    state.blocksSinceLastUpdate = inf;
    state.numAdaptations = 0;
    state.bestNMSE = inf;
end
relDelta = 2;
if isfield(cfg,'adaptation') && isfield(cfg.adaptation,'relNMSE_dB')
    relDelta = cfg.adaptation.relNMSE_dB;
end
% Trigger on degradation with respect to the healthy session reference
% (baselineNMSE, never re-armed): keeps refining after a rupture until the
% performance returns within relDelta of the nominal level or the budget
% is exhausted. Falls back to bestNMSE when no baseline is tracked.
ref = inf;
if isfield(state,'baselineNMSE') && isfinite(state.baselineNMSE)
    ref = state.baselineNMSE;
elseif isfield(state,'bestNMSE') && isfinite(state.bestNMSE)
    ref = state.bestNMSE;
end
% --- Seuil v2 : relDelta est borne par en bas par k ecarts-types du NMSE
% observe sur les blocs sains. Avec un reseau tres bien entraine, la
% reference devient basse et un delta fixe reagit au bruit inter-blocs ;
% le seuil suit alors la dispersion effective du reseau.
if isfield(state,'healthyStd') && isfinite(state.healthyStd) && isfield(cfg.adaptation,'triggerSigma')
    relDelta = max(relDelta, cfg.adaptation.triggerSigma * state.healthyStd);
end
trigger = isfinite(ref) && metrics.NMSE_dB > ref + relDelta;
allowed = state.blocksSinceLastUpdate >= cfg.adaptation.cooldownBlocks && ...
          state.numAdaptations < cfg.adaptation.maxAdaptations;
doAdapt = trigger && allowed;
if doAdapt
    action = onlineFineTunePolicy(metrics, cfg, state);
else
    action = 0;
end
end
