function obs = dqnObservation(metrics, state, cfg)
%DQNOBSERVATION Normalized 6-dim observation for the DQN adaptation agent.
%   Shared by the training environment (createDPDAdaptationEnv) AND the
%   deployment path (evaluateRLPolicy in runDPDScenario) so that the agent
%   sees the SAME state representation during training and use.
%
%   Metrics-only state (no oracle drift): the true PA drift rate is NOT
%   exposed; drift is inferred from the degradation of NMSE vs. the best
%   value observed so far (component 4).
%
%   state fields used: bestNMSE, blocksSinceLastUpdate, numAdaptations, nBlocks.
clip = @(v, lo, hi) min(max(v, lo), hi);

bestN = state.bestNMSE;
if ~isfinite(bestN), bestN = metrics.NMSE_dB; end
nB = 1;
if isfield(state,'nBlocks') && state.nBlocks > 0, nB = state.nBlocks; end
bsu = 0;
if isfield(state,'blocksSinceLastUpdate') && isfinite(state.blocksSinceLastUpdate)
    bsu = state.blocksSinceLastUpdate;
end
maxA = max(cfg.adaptation.maxAdaptations, 1);

obs = [ clip((metrics.NMSE_dB + 14)/14, 0, 1.5);   % 1: NMSE badness (0 = best)
        clip(metrics.EVM_percent/100,   0, 1.5);   % 2: EVM badness
        clip((metrics.ACPR_dBc + 35)/16, 0, 1.5);  % 3: ACPR badness -- recalibrated
        %    for the offset ACPR window (cfg.acpr.offsetFactor = 0.25): the
        %    bench now spans about -37 to -11 dBc, where the previous
        %    (ACPR+25)/21 mapping saturated at 0 below -25 dBc and fed the
        %    agent a constant it had never seen during training.
        clip((metrics.NMSE_dB - bestN)/5, 0, 2);   % 4: drift indicator (>= 0)
        clip(bsu/nB, 0, 1);                        % 5: time since last update
        clip((maxA - state.numAdaptations)/maxA, 0, 1) ]; % 6: budget remaining
end
