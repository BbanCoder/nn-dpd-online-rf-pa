function env = createDPDAdaptationEnv(cfg, baseModel, x)
%CREATEDPDADAPTATIONENV Build a real RL environment for the DPD adaptation MDP.
%
%   env = CREATEDPDADAPTATIONENV(cfg, baseModel, x) returns an rlFunctionEnv
%   (Reinforcement Learning Toolbox required):
%     - Episode : one scenario run over ceil(N/blockSize) blocks.
%     - Reset   : a fresh, randomly-scaled "combined severe" drift realization
%                 (for generalization); the DPD starts from baseModel.
%     - State   : 6-dim normalized vector (see dqnObservation), metrics-only.
%     - Action  : {0 nothing, 1 light FT, 2 strong FT, 3 full recalibration,
%                  4 relax thresholds, 5 tighten thresholds}; executed by
%                 applyPolicyAction, identical to the deployment path.
%     - Reward  : -(w1*NMSE + w2*EVM + w3*ACPR badness) - w4*cost(action),
%                 with w = cfg.rl.weights.
%
%   Without output-arguments baseModel/x this signature cannot run; the
%   metadata-only descriptor previously returned here is superseded.
%
%   baseModel : NN-DPD model trained offline once (trainOfflineNNDPD_ILA).
%   x         : reference waveform (fixed across episodes; drift randomized).

obsInfo = rlNumericSpec([6 1], 'LowerLimit', 0, 'UpperLimit', 2);
obsInfo.Name = 'dpdState';
actInfo = rlFiniteSetSpec([0 1 2 3 4 5]);
actInfo.Name = 'adaptAction';

resetFcn = @() localReset(cfg, baseModel, x);
stepFcn  = @(action, ls) localStep(action, ls);
env = rlFunctionEnv(obsInfo, actInfo, stepFcn, resetFcn);
end

% =========================================================================
function [obs, ls] = localReset(cfg, baseModel, x)
N = numel(x);
% New randomized combined-severe drift realization (severity in [0.7,1.3]).
rng(randi(1e6));
prof = generatePADriftProfile(N, cfg, 'combined');
sev = 0.7 + 0.6*rand;
prof.G           = 1 + sev*(prof.G - 1);
prof.Phi         = sev*prof.Phi;
prof.SatScale    = 1 + sev*(prof.SatScale - 1);
prof.MemoryScale = 1 + sev*(prof.MemoryScale - 1);

ls = struct();
% ACPR useful channel from the measured occupied bandwidth, as in the engine.
cfg.acprBandwidth = measureOccupiedBW(x, cfg.fs);
ls.cfg          = cfg;
ls.x            = x;
ls.N            = N;
ls.dynProfile   = prof;
ls.blockSize    = cfg.adaptation.blockSize;
ls.nBlocks      = ceil(N/cfg.adaptation.blockSize);
ls.blockIdx     = 1;
ls.model        = baseModel;
ls.replayBuffer = struct('input', [], 'target', []);
if isfield(cfg.rl, 'crestSeed')
    % Seed the peak-window store with the waveform crest (engine parity).
    ls.replayBuffer.peakIn = cfg.rl.crestSeed(1);
    ls.replayBuffer.peakTgt = cfg.rl.crestSeed(2);
    ls.replayBuffer.peakVal = cfg.rl.crestSeed{3};
end
ls.numAdaptations = 0;
ls.blocksSinceLastUpdate = 0;
ls.bestNMSE     = inf;

ls = evalBlockLocal(ls);                 % evaluate block 1
obs = dqnObservation(ls.curMetrics, ls2state(ls), ls.cfg);
end

% =========================================================================
function [obs, reward, isDone, ls] = localStep(action, ls)
if iscell(action), action = action{1}; end
action = double(action(1));

% Apply the action to the current block via the shared executor.
budgetLeft = ls.numAdaptations < ls.cfg.adaptation.maxAdaptations;
if action >= 4 || (action > 0 && budgetLeft)
    % Strict ILA pairs (PA output -> PA input z), same as the deployment path.
    [ls.model, ls.cfg, ls.replayBuffer, infoA] = applyPolicyAction( ...
        action, ls.model, ls.curY, ls.curZ, ls.cfg, ls.replayBuffer);
    if infoA.modelUpdated
        ls.numAdaptations = ls.numAdaptations + 1;
        ls.blocksSinceLastUpdate = 0;
        if infoA.action == 3
            ls.bestNMSE = inf;   % re-arm the drift indicator in the new regime
        end
    else
        ls.blocksSinceLastUpdate = ls.blocksSinceLastUpdate + 1;
    end
else
    ls.blocksSinceLastUpdate = ls.blocksSinceLastUpdate + 1;
end

% Advance to the next block and evaluate it with the (possibly updated) model.
ls.blockIdx = ls.blockIdx + 1;
isDone = ls.blockIdx > ls.nBlocks;
if ~isDone
    ls = evalBlockLocal(ls);
end
m = ls.curMetrics;

w = ls.cfg.rl.weights;                   % [wNMSE wEVM wACPR wCost]
clip = @(v, lo, hi) min(max(v, lo), hi);
badness = w(1)*clip((m.NMSE_dB + 14)/14, 0, 1.5) ...
        + w(2)*clip(m.EVM_percent/100,    0, 1.5) ...
        + w(3)*clip((m.ACPR_dBc + 35)/16, 0, 1.5);   % meme echelle que dqnObservation
actionCosts = [0 0.5 1 2 0.1 0.1];       % indexed by action+1
reward = -badness - w(4)*actionCosts(action+1);

obs = dqnObservation(m, ls2state(ls), ls.cfg);
end

% =========================================================================
function ls = evalBlockLocal(ls)
b = ls.blockIdx;
ids = (1 + (b-1)*ls.blockSize):min(b*ls.blockSize, ls.N);
xB = ls.x(ids);
profB = sliceDriftProfileLocal(ls.dynProfile, ids);
zB = applyNNDPD(xB, ls.model, ls.cfg);
yB = paDynamicModel(zB, ls.cfg, profB, ls.cfg.pa.model);
m = struct('NMSE_dB', computeNMSE(yB, xB), ...
           'EVM_percent', computeEVM(yB, xB), ...
           'ACPR_dBc', computeACPR(yB, ls.cfg.fs, ls.cfg.acprBandwidth, ls.cfg.acpr.offsetFactor));
ls.curX = xB;
ls.curZ = zB;
ls.curY = yB;
ls.curMetrics = m;
ls.bestNMSE = min(ls.bestNMSE, m.NMSE_dB);
end

% =========================================================================
function p = sliceDriftProfileLocal(profile, ids)
p = profile;
p.G = profile.G(ids);
p.Phi = profile.Phi(ids);
p.SatScale = profile.SatScale(ids);
p.MemoryScale = profile.MemoryScale(ids,:);
p.DriftRate = profile.DriftRate(ids);
end

% =========================================================================
function s = ls2state(ls)
s.bestNMSE = ls.bestNMSE;
s.blocksSinceLastUpdate = ls.blocksSinceLastUpdate;
s.numAdaptations = ls.numAdaptations;
s.nBlocks = ls.nBlocks;
end
