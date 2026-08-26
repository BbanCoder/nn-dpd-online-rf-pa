function result = runDPDScenario(cfg)
%RUNDPDSCENARIO Common execution engine for S0-S8.
% Covers objectives: HPA drift modeling, dynamic scenarios, DPD comparison, and NMSE/EVM/ACPR evaluation.
rng(cfg.rngSeed);
ticScenario = tic;
ensureDirsLocal(cfg);

if strcmpi(cfg.waveform,'5GNR')
    [x, info] = generate5GNRWaveform(cfg);
else
    [x, info] = generateQAMOFDMWaveform(cfg);
end
x = normalizeSignal(x);
info.IBO_dB = 10*log10(cfg.pa.satLevel^2 / mean(abs(x).^2));
info.PAPR_dB = computePAPR(x);
% ACPR useful channel = measured 99% occupied bandwidth of the reference
% signal (configuration labels may not match the actual waveform).
cfg.acprBandwidth = measureOccupiedBW(x, cfg.fs);
info.OccupiedBW_MHz = cfg.acprBandwidth / 1e6;

N = numel(x);
paType = upper(cfg.pa.model);  % 'MP' (default) or 'GMP' via cfg.pa.model
nominalProfile = generatePADriftProfile(N, cfg, 'nominal');
dynProfile = generatePADriftProfile(N, cfg, cfg.scenario.driftType);
yNom = paDynamicModel(x, cfg, nominalProfile, paType);
yNo = paDynamicModel(x, cfg, dynProfile, paType);

% Save drift profiles for methodology traceability.
try
    save(fullfile(cfg.paths.mat,'paDriftProfiles.mat'), 'nominalProfile', 'dynProfile');
catch ME
    warning('Could not save PA drift profiles: %s', ME.message);
end

% Classical ILA DPD training under nominal PA.
mpModel = trainMPDPD(yNom, x, cfg);
gmpModel = trainGMPDPD(yNom, x, cfg);
zMP = applyMPDPD(x, mpModel);
zGMP = applyGMPDPD(x, gmpModel);
yMP = paDynamicModel(zMP, cfg, dynProfile, paType);
yGMP = paDynamicModel(zGMP, cfg, dynProfile, paType);

methods = struct();
methods.No_DPD = yNo;
methods.MP_DPD = yMP;
methods.GMP_DPD = yGMP;

% Three evaluation periods: before / during / after the drift.
iBefore = 1:floor(N/3);
iDuring = floor(N/3)+1:floor(2*N/3);
iAfter = ceil(2*N/3):N;

summary = table();
summary = [summary; summaryRowLocal(cfg, info, 'No_DPD', yNo, x, iBefore, iDuring, iAfter, 0, toc(ticScenario), cfg.scenario.comment)];
summary = [summary; summaryRowLocal(cfg, info, 'MP_DPD', yMP, x, iBefore, iDuring, iAfter, 0, toc(ticScenario), 'Static MP-DPD baseline')];
summary = [summary; summaryRowLocal(cfg, info, 'GMP_DPD', yGMP, x, iBefore, iDuring, iAfter, 0, toc(ticScenario), 'Static GMP-DPD baseline')];

nnModel = [];
yNNStatic = [];
yNNAdapt = [];
numAdapt = 0;
timeline = emptyTimelineLocal();
adaptInfo = struct([]);
replayBuffer = struct('input',[],'target',[]);

if cfg.nn.enabled && isfield(cfg,'toolboxes') && cfg.toolboxes.deepLearning
    nnModel = trainOfflineNNDPD_ILA(yNom, x, cfg);
    zNNStatic = applyNNDPD(x, nnModel, cfg);
    yNNStatic = paDynamicModel(zNNStatic, cfg, dynProfile, paType);
    methods.NN_DPD_Static = yNNStatic;
    summary = [summary; summaryRowLocal(cfg, info, 'NN_DPD_Static', yNNStatic, x, iBefore, iDuring, iAfter, 0, toc(ticScenario), 'Offline NN-DPD with fixed weights')];

    currentModel = nnModel;
    % Seed ONLY the peak-window store with the global waveform crest (from
    % the nominal pairs): online updates otherwise never see crest-level
    % amplitudes and the network response drifts in that region. The
    % uniform replay buffer is NOT seeded — bulk nominal pairs would bias
    % post-drift updates toward the outdated PA state.
    [~, crestIdx] = max(abs(x));
    lo = max(1, crestIdx-64); hi = min(N, crestIdx+63);
    replayBuffer.peakIn = {yNom(lo:hi)};
    replayBuffer.peakTgt = {x(lo:hi)};
    replayBuffer.peakVal = max(abs(x));
    blockSize = cfg.adaptation.blockSize;
    nBlocks = ceil(N/blockSize);
    yNNAdapt = zeros(N,1);
    state.blocksSinceLastUpdate = inf;
    state.numAdaptations = 0;
    state.bestNMSE = inf;       % re-armed after each full recalibration (action grading)
    state.baselineNMSE = inf;   % never re-armed: healthy session reference (trigger)
    state.nBlocks = nBlocks;
    rlAgent = [];
    if isfield(cfg.scenario,'useRL') && cfg.scenario.useRL
        rlAgent = trainDQNAdaptationAgent(cfg);
    end

    for b = 1:nBlocks
        ids = (1+(b-1)*blockSize):min(b*blockSize,N);
        xBlock = x(ids);
        profBlock = sliceProfileLocal(dynProfile, ids);
        zBlock = applyNNDPD(xBlock, currentModel, cfg);
        yBlock = paDynamicModel(zBlock, cfg, profBlock, paType);
        yNNAdapt(ids) = yBlock;
        mAdapt = computeMetricsLocal(yBlock, xBlock, cfg);
        if ~isempty(yNNStatic)
            mStatic = computeMetricsLocal(yNNStatic(ids), xBlock, cfg);
        else
            mStatic = mAdapt;
        end
        state.bestNMSE = min(state.bestNMSE, mAdapt.NMSE_dB);
        state.baselineNMSE = min(state.baselineNMSE, mAdapt.NMSE_dB);
        timeline.block(end+1) = b;
        timeline.NMSE_static(end+1) = mStatic.NMSE_dB;
        timeline.NMSE_adapt(end+1) = mAdapt.NMSE_dB;
        timeline.EVM_static(end+1) = mStatic.EVM_percent;
        timeline.EVM_adapt(end+1) = mAdapt.EVM_percent;
        timeline.ACPR_static(end+1) = mStatic.ACPR_dBc;
        timeline.ACPR_adapt(end+1) = mAdapt.ACPR_dBc;
        timeline.adaptEvent(end+1) = 0;

        if numel(xBlock) >= cfg.adaptation.minSamples
            [doAdapt, action] = adaptationScheduler(mAdapt, cfg, state);
            if isfield(cfg.scenario,'useRL') && cfg.scenario.useRL
                action = evaluateRLPolicy(rlAgent, mAdapt, state, cfg);
                % Threshold actions (4-5) are always allowed; model updates
                % (1-3) are limited by the adaptation budget.
                doAdapt = action >= 4 || (action > 0 && state.numAdaptations < cfg.adaptation.maxAdaptations);
            end
            if doAdapt && action > 0
                % Strict ILA pairs (PA output -> PA input z): when the chain is
                % already linearized y ~ K*x, so (y,x) pairs degenerate to a
                % linear map and erase the predistortion; (y,z) never does.
                [currentModel, cfg, replayBuffer, infoUpdate] = applyPolicyAction( ...
                    action, currentModel, yBlock, zBlock, cfg, replayBuffer);
                adaptInfo = [adaptInfo; infoUpdate]; %#ok<AGROW>
                if infoUpdate.modelUpdated
                    numAdapt = numAdapt + 1;
                    state.numAdaptations = numAdapt;
                    state.blocksSinceLastUpdate = 0;
                    timeline.adaptEvent(end) = 1;
                    if infoUpdate.action == 3
                        % Re-arm the jump detector in the new PA regime,
                        % otherwise the stale pre-rupture best NMSE keeps
                        % re-triggering full recalibrations.
                        state.bestNMSE = inf;
                    end
                else
                    state.blocksSinceLastUpdate = state.blocksSinceLastUpdate + 1;
                end
            else
                state.blocksSinceLastUpdate = state.blocksSinceLastUpdate + 1;
            end
        end
    end
    methods.NN_DPD_Adaptive = yNNAdapt;
    methodName = 'NN_DPD_Adaptive';
    if isfield(cfg.scenario,'useRL') && cfg.scenario.useRL
        methodName = 'RL_Adaptive_Policy';
    end
    summary = [summary; summaryRowLocal(cfg, info, methodName, yNNAdapt, x, iBefore, iDuring, iAfter, numAdapt, toc(ticScenario), 'Online NN-DPD adaptation')];
else
    fprintf('[INFO] Deep Learning Toolbox unavailable or NN disabled. NN-DPD methods skipped for %s.\n', cfg.scenario.id);
end

% Figures (skipped when cfg.plots.savePNG is false, e.g. Monte-Carlo runs)
try
    if ~isfield(cfg.plots,'savePNG') || cfg.plots.savePNG
    plotPSDComparison(methods, cfg.fs, cfg.scenario.id, cfg);
    plotAMAM_AMPM(x, yNom, yNo, cfg.scenario.id, cfg);
    plotConstellation(methods, cfg.scenario.id, cfg, info);
    plotMetricTimeline(timeline, cfg.scenario.id, cfg);
    plotAdaptationEvents(timeline, cfg.scenario.id, cfg);
    end
catch ME
    warning('Plot generation failed for %s: %s', cfg.scenario.id, ME.message);
end

% Scenario-specific MAT save.
result = struct();
result.id = cfg.scenario.id;
result.name = cfg.scenario.name;
result.info = info;
result.summary = summary;
result.timeline = timeline;
result.adaptInfo = adaptInfo;
result.metrics = methods;
result.comment = cfg.scenario.comment;
try
    save(fullfile(cfg.paths.mat, sprintf('%s_results.mat', cfg.scenario.id)), 'result', '-v7.3');
catch ME
    warning('Could not save scenario MAT file: %s', ME.message);
end
end

function row = summaryRowLocal(cfg, info, methodName, y, x, iBefore, iDuring, iAfter, numAdapt, runtime_s, comment)
mB = computeMetricsLocal(y(iBefore), x(iBefore), cfg);
mD = computeMetricsLocal(y(iDuring), x(iDuring), cfg);
mA = computeMetricsLocal(y(iAfter), x(iAfter), cfg);
row = summarizeMetrics(cfg.scenario.id, info, methodName, mB, mD, mA, numAdapt, runtime_s, comment);
end

function m = computeMetricsLocal(y, ref, cfg)
m = struct();
m.NMSE_dB = computeNMSE(y, ref);
m.EVM_percent = computeEVM(y, ref);
if isfield(cfg, 'acprBandwidth')
    acprBW = cfg.acprBandwidth;
else
    acprBW = cfg.bandwidth;
end
offFac = 0;
if isfield(cfg, 'acpr') && isfield(cfg.acpr, 'offsetFactor')
    offFac = cfg.acpr.offsetFactor;
end
m.ACPR_dBc = computeACPR(y, cfg.fs, acprBW, offFac);
end

function tl = emptyTimelineLocal()
tl = struct('block',[],'NMSE_static',[],'NMSE_adapt',[], ...
            'EVM_static',[],'EVM_adapt',[], 'ACPR_static',[], 'ACPR_adapt',[], 'adaptEvent',[]);
end

function p = sliceProfileLocal(profile, ids)
p = profile;
p.G = profile.G(ids);
p.Phi = profile.Phi(ids);
p.SatScale = profile.SatScale(ids);
p.MemoryScale = profile.MemoryScale(ids,:);
p.DriftRate = profile.DriftRate(ids);
end

function ensureDirsLocal(cfg)
for d = {cfg.paths.figures, cfg.paths.tables, cfg.paths.mat}
    if ~exist(d{1}, 'dir'), mkdir(d{1}); end
end
end
