function [rawTable, statsTable] = run_montecarlo(seeds, scenarioIds)
%RUN_MONTECARLO Multi-seed Monte-Carlo evaluation of selected scenarios.
%
%   run_montecarlo                     % seeds 1:10, scenarios {'S2','S6','S8'}
%   run_montecarlo(1:5, {'S0','S2'})   % custom seeds and scenarios
%
%   For each seed, the full chain (waveform, PA, DPD training, drift,
%   adaptation) is re-run with cfg.rngSeed = seed, giving an independent
%   realization of the OFDM data, of the NN initialization and of the
%   adaptation trajectory. Outputs are redirected to results/montecarlo/
%   so the reference seed-42 figures and tables in results/ are preserved;
%   per-scenario figures are skipped to save time.
%
%   Outputs:
%     results/montecarlo/montecarlo_raw.csv     one row per seed/scenario/method
%     results/montecarlo/montecarlo_summary.csv mean/std per scenario/method
%
%   NOTE (S8): if rl_policy/dqn_agent.mat exists, S8 loads that DQN agent;
%   otherwise the heuristic policy is used. Do NOT include S8 while a DQN
%   training run has not yet finished writing dqn_agent.mat, and state in
%   the thesis which policy the S8 rows correspond to.

if nargin < 1 || isempty(seeds), seeds = 1:10; end
if nargin < 2 || isempty(scenarioIds), scenarioIds = {'S2','S6','S8'}; end
if ~iscell(scenarioIds), scenarioIds = cellstr(scenarioIds); end

projectRoot = fileparts(mfilename('fullpath'));
addpath(genpath(projectRoot));

fns = struct('S0', @scenario_S0_NominalCalibration, ...
             'S1', @scenario_S1_ThermalDrift, ...
             'S2', @scenario_S2_PowerJump, ...
             'S3', @scenario_S3_MemoryVariation, ...
             'S4', @scenario_S4_AgingDrift, ...
             'S5', @scenario_S5_LoadMismatch, ...
             'S6', @scenario_S6_CombinedDynamic, ...
             'S7', @scenario_S7_5GStressTest, ...
             'S8', @scenario_S8_RLAdaptation);

mcRoot = fullfile(projectRoot, 'results', 'montecarlo');
if ~exist(mcRoot, 'dir'), mkdir(mcRoot); end

ticAll = tic;
rawTable = table();
for s = seeds(:).'
    cfg = defaultConfig(projectRoot);
    cfg.rngSeed = s;
    cfg.toolboxes = toolboxCheck();
    % Redirect all outputs away from the reference results/ tree.
    cfg.paths.figures = fullfile(mcRoot, 'figures');
    cfg.paths.tables = mcRoot;
    cfg.paths.mat = fullfile(mcRoot, 'mat');
    cfg.plots.savePNG = false;   % skip per-scenario figures (time saver)
    for k = 1:numel(scenarioIds)
        id = upper(char(scenarioIds{k}));
        if ~isfield(fns, id)
            warning('[MC] Unknown scenario id %s, skipped.', id);
            continue;
        end
        fprintf('[MC] seed %d, scenario %s ... ', s, id);
        tScn = tic;
        try
            res = fns.(id)(cfg);
            rows = res.summary;
            rows.Seed = repmat(s, height(rows), 1);
            rawTable = [rawTable; rows]; %#ok<AGROW>
            fprintf('ok (%.1f s)\n', toc(tScn));
        catch ME
            warning('[MC] seed %d scenario %s failed: %s', s, id, ME.message);
        end
    end
end

if isempty(rawTable)
    error('Monte-Carlo produced no results.');
end

writetable(rawTable, fullfile(mcRoot, 'montecarlo_raw.csv'));

statsTable = groupsummary(rawTable, {'Scenario','Method'}, {'mean','std'}, ...
    {'NMSE_dB_before','NMSE_dB_during','NMSE_dB_after', ...
     'EVM_percent_after','ACPR_dBc_after','NumAdaptations'});
writetable(statsTable, fullfile(mcRoot, 'montecarlo_summary.csv'));

fprintf('\nMonte-Carlo terminé en %.1f min : %d graines x %d scénarios.\n', ...
    toc(ticAll)/60, numel(seeds), numel(scenarioIds));
fprintf('Résultats :\n  %s\n  %s\n', ...
    fullfile(mcRoot,'montecarlo_raw.csv'), fullfile(mcRoot,'montecarlo_summary.csv'));
end
