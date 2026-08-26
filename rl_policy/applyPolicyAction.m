function [model, cfg, replayBuffer, info] = applyPolicyAction(action, model, paOutput, reference, cfg, replayBuffer)
%APPLYPOLICYACTION Execute one adaptation-policy action.
%   Shared by the scenario engine (runDPDScenario) and the RL training
%   environment (createDPDAdaptationEnv) so that a given action has exactly
%   the same effect during training and deployment.
%
%   Action space:
%     0 : do nothing
%     1 : light fine-tuning  (2 epochs, nominal online learning rate)
%     2 : strong fine-tuning (5 epochs, doubled online learning rate)
%     3 : full recalibration (warm-start retraining with the offline budget
%         and learning rate on the recent block, replay buffer reset)
%     4 : relax adaptation thresholds  (NMSE +2 dB, EVM x1.25, bounded)
%     5 : tighten adaptation thresholds (NMSE -2 dB, EVM /1.25, bounded)
%
%   info.modelUpdated is true for actions 1-3 (counted as adaptations);
%   actions 4-5 only modify cfg.thresholds.
info = struct('action',action,'modelUpdated',false,'thresholdsChanged',false, ...
              'lossBefore',NaN,'lossAfter',NaN,'iterations',0,'runtime_s',0,'reverted',false);
switch action
    case 0
        return;
    case 1
        cfgU = cfg;
        cfgU.adaptation.maxEpochsOnline = 2;
        [model, up, replayBuffer] = updateOnlineNNDPD(model, paOutput, reference, cfgU, replayBuffer);
        info = mergeUpdateInfoLocal(info, up);
    case 2
        cfgU = cfg;
        cfgU.adaptation.maxEpochsOnline = 5;
        cfgU.adaptation.learningRateOnline = 2*cfg.adaptation.learningRateOnline;
        [model, up, replayBuffer] = updateOnlineNNDPD(model, paOutput, reference, cfgU, replayBuffer);
        info = mergeUpdateInfoLocal(info, up);
    case 3
        % Full recalibration: warm-start from the current weights on fresh
        % data only (replay reset first, discarding stale pre-rupture pairs)
        % with the offline epoch budget but the ONLINE learning rate.
        % Retraining from scratch, or at the offline learning rate, on a
        % single block is severely under-determined and destroys the model.
        replayBuffer = struct('input',[],'target',[]);
        cfgU = cfg;
        cfgU.adaptation.maxEpochsOnline = cfg.nn.maxEpochsOffline;
        [model, up, replayBuffer] = updateOnlineNNDPD(model, paOutput, reference, cfgU, replayBuffer);
        info = mergeUpdateInfoLocal(info, up);
    case 4
        cfg.thresholds.NMSE_dB = min(cfg.thresholds.NMSE_dB + 2, -10);
        cfg.thresholds.EVM_percent = min(cfg.thresholds.EVM_percent*1.25, 20);
        info.thresholdsChanged = true;
    case 5
        cfg.thresholds.NMSE_dB = max(cfg.thresholds.NMSE_dB - 2, -40);
        cfg.thresholds.EVM_percent = max(cfg.thresholds.EVM_percent/1.25, 1);
        info.thresholdsChanged = true;
    otherwise
        error('Unknown policy action: %g', action);
end
end

function info = mergeUpdateInfoLocal(info, up)
info.modelUpdated = true;
info.lossBefore = up.lossBefore;
info.lossAfter = up.lossAfter;
info.iterations = up.iterations;
info.runtime_s = up.runtime_s;
info.reverted = up.reverted;
end
