function plotScenarioSummary(summaryTable, cfg)
%PLOTSCENARIOSUMMARY Save global bar summary for final NMSE by method.
%   Bars are grouped by scenario with one bar per method, so the axis
%   carries one label per scenario instead of one per scenario-method
%   pair (45 labels overlapped and could not be read).
if isempty(summaryTable) || ~ismember('NMSE_dB_after', summaryTable.Properties.VariableNames)
    return;
end
valid = isfinite(summaryTable.NMSE_dB_after);
if ~any(valid)
    return;
end
T = summaryTable(valid, :);

scenarios = unique(string(T.Scenario), 'stable');
methods   = unique(string(T.Method), 'stable');

% Pivot to scenarios x methods; NaN marks a pair that was never run.
values = nan(numel(scenarios), numel(methods));
for k = 1:height(T)
    i = find(scenarios == string(T.Scenario(k)), 1);
    j = find(methods   == string(T.Method(k)),   1);
    values(i, j) = T.NMSE_dB_after(k);
end

% Drop methods with no finite result so the legend stays honest.
keep = any(isfinite(values), 1);
values  = values(:, keep);
methods = methods(keep);

fig = figure('Visible', cfg.plots.visible, 'Position', [100 100 1150 520]);
b = bar(categorical(scenarios, scenarios), values, 'grouped');
grid on;
ylabel('NMSE after (dB)'); xlabel('Scenario');
title('Scenario summary - final NMSE');
legend(b, strrep(methods, '_', ' '), 'Location', 'eastoutside', 'Interpreter', 'none');
saveas(fig, fullfile(cfg.paths.figures, 'Global_summary_NMSE.png'));
close(fig);
end
