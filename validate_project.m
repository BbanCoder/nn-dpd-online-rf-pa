clear; clc; close all;
rng(42);
projectRoot = fileparts(mfilename('fullpath'));
addpath(genpath(projectRoot));

cfg = defaultConfig(projectRoot);
cfg.toolboxes = toolboxCheck();
ensureProjectDirsLocal(cfg);
report = {};

report{end+1} = sprintf('Validation started: %s', datestr(now));
report{end+1} = sprintf('Project root: %s', projectRoot);

expectedFiles = {
    'run_all.m','validate_project.m','README.md', ...
    'config/defaultConfig.m','config/scenarioConfig.m','config/toolboxCheck.m', ...
    'waveforms/generate5GNRWaveform.m','waveforms/generateQAMOFDMWaveform.m','waveforms/normalizeSignal.m','waveforms/computePAPR.m', ...
    'pa_models/paMemorylessRapp.m','pa_models/paMemorylessSaleh.m','pa_models/paMemoryPolynomial.m','pa_models/paGMP.m','pa_models/paDynamicModel.m','pa_models/generatePADriftProfile.m', ...
    'dpd_classical/buildMPRegressor.m','dpd_classical/trainMPDPD.m','dpd_classical/applyMPDPD.m','dpd_classical/buildGMPRegressor.m','dpd_classical/trainGMPDPD.m','dpd_classical/applyGMPDPD.m', ...
    'dpd_neural/createRVTDNN.m','dpd_neural/createMLPNNDPD.m','dpd_neural/createGRUNNDPD.m','dpd_neural/makeNNFeatures.m','dpd_neural/trainOfflineNNDPD_ILA.m','dpd_neural/trainOfflineNNDPD_DLA.m','dpd_neural/applyNNDPD.m','dpd_neural/updateOnlineNNDPD.m','dpd_neural/replayBufferUpdate.m','dpd_neural/computeForgettingIndex.m', ...
    'adaptation/driftDetectorNMSE.m','adaptation/driftDetectorEVM.m','adaptation/adaptationScheduler.m','adaptation/onlineFineTunePolicy.m','adaptation/periodicUpdatePolicy.m', ...
    'rl_policy/createDPDAdaptationEnv.m','rl_policy/trainDQNAdaptationAgent.m','rl_policy/evaluateRLPolicy.m', ...
    'metrics/computeNMSE.m','metrics/computeEVM.m','metrics/computeACPR.m','metrics/computeBER.m','metrics/computePAEProxy.m','metrics/summarizeMetrics.m', ...
    'scenarios/scenario_S0_NominalCalibration.m','scenarios/scenario_S1_ThermalDrift.m','scenarios/scenario_S2_PowerJump.m','scenarios/scenario_S3_MemoryVariation.m','scenarios/scenario_S4_AgingDrift.m','scenarios/scenario_S5_LoadMismatch.m','scenarios/scenario_S6_CombinedDynamic.m','scenarios/scenario_S7_5GStressTest.m','scenarios/scenario_S8_RLAdaptation.m','scenarios/runDPDScenario.m', ...
    'plots/plotPSDComparison.m','plots/plotAMAM_AMPM.m','plots/plotConstellation.m','plots/plotMetricTimeline.m','plots/plotAdaptationEvents.m','plots/plotScenarioSummary.m'};

missing = {};
for i = 1:numel(expectedFiles)
    if ~exist(fullfile(projectRoot, expectedFiles{i}), 'file')
        missing{end+1} = expectedFiles{i}; %#ok<SAGROW>
    end
end
if isempty(missing)
    report{end+1} = '[OK] All expected files exist.';
else
    report{end+1} = sprintf('[FAIL] Missing files: %s', strjoin(missing, ', '));
end

try
    [x, info] = generateQAMOFDMWaveform(cfg);
    x = normalizeSignal(x);
    assert(~isempty(x) && all(isfinite(real(x))) && all(isfinite(imag(x))), 'Invalid generated signal');
    pwr = mean(abs(x).^2);
    assert(abs(pwr - 1) < 1e-8, 'Power normalization failed');
    report{end+1} = sprintf('[OK] Waveform generation and normalization. Pavg = %.6f, PAPR = %.2f dB.', pwr, computePAPR(x));
catch ME
    report{end+1} = sprintf('[FAIL] Waveform test: %s', ME.message);
end

try
    prof = generatePADriftProfile(numel(x), cfg, 'nominal');
    y = paDynamicModel(x, cfg, prof, 'MP');
    assert(numel(y) == numel(x), 'PA output length mismatch');
    assert(all(isfinite(real(y))) && all(isfinite(imag(y))), 'PA output contains NaN/Inf');
    nmseDb = computeNMSE(y, x);
    evmPct = computeEVM(y, x);
    acprDb = computeACPR(y, cfg.fs, cfg.bandwidth, cfg.acpr.offsetFactor);
    report{end+1} = sprintf('[OK] Metrics test. NMSE=%.2f dB, EVM=%.2f%%, ACPR=%.2f dBc.', nmseDb, evmPct, acprDb);
catch ME
    report{end+1} = sprintf('[FAIL] PA/metrics test: %s', ME.message);
end

try
    c = scenarioConfig(cfg, 'S0');
    c.fastValidation = true;
    res = scenario_S0_NominalCalibration(c);
    report{end+1} = '[OK] S0 scenario executed in validation mode.';
    if any(res.summary.Method == "MP_DPD") || any(res.summary.Method == "GMP_DPD")
        report{end+1} = '[OK] S0 produced DPD baseline results.';
    end
catch ME
    report{end+1} = sprintf('[FAIL] S0 validation scenario: %s', ME.message);
end

try
    executeRunAllLocal(projectRoot);
    report{end+1} = '[OK] run_all.m executed without blocking error.';
catch ME
    report{end+1} = sprintf('[FAIL] run_all.m execution: %s', ME.message);
end

requiredOutputs = {fullfile(cfg.paths.tables,'summary_all_scenarios.csv'), ...
                   fullfile(cfg.paths.mat,'allResults.mat')};
for i = 1:numel(requiredOutputs)
    if exist(requiredOutputs{i}, 'file')
        report{end+1} = sprintf('[OK] Output exists: %s', requiredOutputs{i});
    else
        report{end+1} = sprintf('[FAIL] Output missing: %s', requiredOutputs{i});
    end
end
if exist(fullfile(cfg.paths.tables,'summary_all_scenarios.xlsx'), 'file')
    report{end+1} = '[OK] XLSX summary exists.';
else
    report{end+1} = '[WARN] XLSX summary not found; CSV may still be valid.';
end

reportFile = fullfile(cfg.paths.tables, 'validation_report.txt');
fid = fopen(reportFile, 'w');
for i = 1:numel(report)
    fprintf(fid, '%s\n', report{i});
    fprintf('%s\n', report{i});
end
fclose(fid);
fprintf('\nValidation report written to %s\n', reportFile);

function executeRunAllLocal(projectRoot)
    run(fullfile(projectRoot, 'run_all.m'));
end

function ensureProjectDirsLocal(cfg)
    dirs = {cfg.paths.figures, cfg.paths.tables, cfg.paths.mat};
    for d = 1:numel(dirs)
        if ~exist(dirs{d}, 'dir')
            mkdir(dirs{d});
        end
    end
end
