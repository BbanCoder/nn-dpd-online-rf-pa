function action = evaluateRLPolicy(agent, metrics, state, cfg)
%EVALUATERLPOLICY Choose an adaptation action.
%   action = EVALUATERLPOLICY(agent, metrics, state, cfg)
%     - If AGENT is a trained RL agent object (rlDQNAgent), builds the
%       normalized observation (dqnObservation) and queries getAction.
%     - Otherwise (empty, or a struct heuristic descriptor), falls back to
%       the heuristic onlineFineTunePolicy on the raw metrics and state.
%
%   metrics : struct with NMSE_dB, EVM_percent, ACPR_dBc (raw units).
%   state   : struct with bestNMSE, blocksSinceLastUpdate, numAdaptations, nBlocks.
isRealAgent = ~isempty(agent) && ~isstruct(agent);
if ~isRealAgent
    action = onlineFineTunePolicy(metrics, cfg, state);
    return;
end
try
    obs = dqnObservation(metrics, state, cfg);
    a = getAction(agent, obs);
    if iscell(a), a = a{1}; end
    action = double(a(1));
catch
    action = onlineFineTunePolicy(metrics, cfg, state);   % robust fallback
end
end
