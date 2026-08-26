clear; clc; close all;
rng(42);
projectRoot = fileparts(mfilename('fullpath'));
addpath(genpath(projectRoot));

cfg = defaultConfig(projectRoot);
cfg.toolboxes = toolboxCheck();
ensureProjectDirs(cfg);

scenarioFns = {@scenario_S0_NominalCalibration, ...
               @scenario_S1_ThermalDrift, ...
               @scenario_S2_PowerJump, ...
               @scenario_S3_MemoryVariation, ...
               @scenario_S4_AgingDrift, ...
               @scenario_S5_LoadMismatch, ...
               @scenario_S6_CombinedDynamic, ...
               @scenario_S7_5GStressTest, ...
               @scenario_S8_RLAdaptation};

allResults = struct();
summaryTable = table();

fprintf('\n=== NN-DPD Online RF PA Master Project ===\n');
fprintf('MATLAB project root: %s\n', projectRoot);

for k = 1:numel(scenarioFns)
    try
        scenarioResult = scenarioFns{k}(cfg);
        allResults.(scenarioResult.id) = scenarioResult;
        if isfield(scenarioResult,'summary') && ~isempty(scenarioResult.summary)
            summaryTable = [summaryTable; scenarioResult.summary]; %#ok<AGROW>
        end
        fprintf('[OK] %s - %s\n', scenarioResult.id, scenarioResult.name);
    catch ME
        warning('[FAILED] Scenario %d: %s', k, ME.message);
        failRow = makeFailureRow(sprintf('S%d',k-1), ME.message);
        summaryTable = [summaryTable; failRow]; %#ok<AGROW>
    end
end

if isempty(summaryTable)
    summaryTable = makeFailureRow('NONE','No scenario produced results');
end

csvFile = fullfile(cfg.paths.tables,'summary_all_scenarios.csv');
xlsxFile = fullfile(cfg.paths.tables,'summary_all_scenarios.xlsx');
matFile = fullfile(cfg.paths.mat,'allResults.mat');

writetable(summaryTable, csvFile);
try
    writetable(summaryTable, xlsxFile);
catch ME
    warning('Could not write XLSX file: %s. CSV file was written.', ME.message);
end
save(matFile, 'allResults', 'summaryTable', 'cfg', '-v7.3');

try
    plotScenarioSummary(summaryTable, cfg);
catch ME
    warning('Could not generate global summary figure: %s', ME.message);
end

for metricName = ["NMSE", "EVM", "ACPR"]
    for periodName = ["during", "after"]
        try
            plotAdaptationGain(summaryTable, cfg, char(metricName), char(periodName));
        catch ME
            warning('Could not generate %s %s adaptation gain figure: %s', ...
                metricName, periodName, ME.message);
        end
    end
end

fprintf('\nResults written to:\n  %s\n  %s\n  %s\n', csvFile, xlsxFile, matFile);

function ensureProjectDirs(cfg)
    dirs = {cfg.paths.figures, cfg.paths.tables, cfg.paths.mat};
    for i = 1:numel(dirs)
        if ~exist(dirs{i}, 'dir')
            mkdir(dirs{i});
        end
    end
end

function row = makeFailureRow(scenarioId, msg)
    row = table(string(scenarioId), "N/A", "N/A", NaN, NaN, "FAILED", ...
        NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 0, NaN, string(msg), ...
        'VariableNames', {'Scenario','Waveform','Modulation','Bandwidth_MHz','IBO_dB','Method', ...
        'NMSE_dB_before','NMSE_dB_during','NMSE_dB_after', ...
        'EVM_percent_before','EVM_percent_during','EVM_percent_after', ...
        'ACPR_dBc_before','ACPR_dBc_during','ACPR_dBc_after', ...
        'NumAdaptations','Runtime_s','Comments'});
end
