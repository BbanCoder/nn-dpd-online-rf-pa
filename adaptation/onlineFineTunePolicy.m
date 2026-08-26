function action = onlineFineTunePolicy(metrics, cfg, state)
%ONLINEFINETUNEPOLICY Heuristic action graded on the relative NMSE degradation.
%   0 = do nothing, 1 = light fine-tuning, 2 = strong fine-tuning,
%   3 = full recalibration. The intensity is tiered on the degradation of
%   the block NMSE with respect to the best (healthy) NMSE observed so far:
%   >8 dB (abrupt rupture) -> 3, >4 dB -> 2, >relNMSE_dB (default 2) -> 1.
%   Without a STATE argument (bestNMSE unknown), falls back to the absolute
%   thresholds and is limited to actions 0-2.
relDelta = 2;
if isfield(cfg,'adaptation') && isfield(cfg.adaptation,'relNMSE_dB')
    relDelta = cfg.adaptation.relNMSE_dB;
end
if nargin >= 3 && isstruct(state) && isfield(state,'bestNMSE') && isfinite(state.bestNMSE)
    % The heuristic is capped at strong fine-tuning: in every tested drift
    % case the full recalibration (action 3) destroyed more than it fixed
    % when triggered blindly on a degradation threshold. Action 3 remains
    % in the action space for the RL agent, which can learn WHEN it pays.
    nmseJump = metrics.NMSE_dB - state.bestNMSE;
    if nmseJump > 4
        action = 2; % strong fine-tuning
    elseif nmseJump > relDelta
        action = 1; % light fine-tuning
    else
        action = 0;
    end
else
    % Legacy absolute-threshold fallback (no healthy reference available).
    if metrics.EVM_percent > 2*cfg.thresholds.EVM_percent || metrics.NMSE_dB > cfg.thresholds.NMSE_dB + 8
        action = 2;
    elseif metrics.NMSE_dB > cfg.thresholds.NMSE_dB || metrics.EVM_percent > cfg.thresholds.EVM_percent
        action = 1;
    else
        action = 0;
    end
end
end
