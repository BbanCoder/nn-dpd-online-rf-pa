function plotAdaptationGain(summaryTable, cfg, metric, period)
%PLOTADAPTATIONGAIN Save per-scenario gain of NN_DPD_Adaptive over NN_DPD_Static.
%   The bar height is (static - adaptive) for the chosen metric, so a
%   positive bar always means online adaptation helped, whatever the sign
%   convention of the metric itself. All three metrics improve when they
%   decrease: NMSE (dB), EVM (%) and ACPR (dBc).
%
%   metric is 'NMSE' (default), 'EVM' or 'ACPR'.
%   period is 'after' (default), 'during' or 'before'.
%
%   These bars come from the single reference run. Where a Monte-Carlo
%   study exists (S2, S6, S8), it supersedes them: a per-seed spread of
%   several tenths of a dB means one seed cannot settle the sign of a
%   gain this small. The subtitle states this on the figure itself.
if nargin < 3 || isempty(metric), metric = 'NMSE'; end
if nargin < 4 || isempty(period), period = 'after'; end

switch upper(metric)
    case 'NMSE', stem = 'NMSE_dB';     unit = 'dB';     label = 'NMSE';
    case 'EVM',  stem = 'EVM_percent'; unit = 'points'; label = 'EVM';
    case 'ACPR', stem = 'ACPR_dBc';    unit = 'dB';     label = 'ACPR';
    otherwise, error('plotAdaptationGain:badMetric', 'Unknown metric "%s".', metric);
end
switch lower(period)
    case 'after',  periodLabel = 'après la dérive';
    case 'during', periodLabel = 'pendant la dérive';
    case 'before', periodLabel = 'avant la dérive';
    otherwise, error('plotAdaptationGain:badPeriod', 'Unknown period "%s".', period);
end
col = sprintf('%s_%s', stem, lower(period));
if isempty(summaryTable) || ~ismember(col, summaryTable.Properties.VariableNames)
    return;
end

scenarios = unique(string(summaryTable.Scenario), 'stable');
gain = nan(numel(scenarios), 1);
for i = 1:numel(scenarios)
    inScenario = string(summaryTable.Scenario) == scenarios(i);
    s = summaryTable{inScenario & string(summaryTable.Method) == "NN_DPD_Static",   col};
    a = summaryTable{inScenario & string(summaryTable.Method) == "NN_DPD_Adaptive", col};
    if isscalar(s) && isscalar(a) && isfinite(s) && isfinite(a)
        gain(i) = s - a;
    end
end
keep = isfinite(gain);
if ~any(keep), return; end
scenarios = scenarios(keep);
gain      = gain(keep);

fig = figure('Visible', cfg.plots.visible, 'Position', [100 100 900 480]);
hold on; grid on;
cats = categorical(scenarios, scenarios);
% Two series so the sign reads without checking the axis.
bar(cats, max(gain, 0), 0.6, 'FaceColor', [0.20 0.60 0.30], 'DisplayName', 'Adaptation gagnante');
bar(cats, min(gain, 0), 0.6, 'FaceColor', [0.80 0.33 0.25], 'DisplayName', 'Adaptation perdante');
yline(0, 'k-', 'LineWidth', 1, 'HandleVisibility', 'off');

offset = 0.06 * max(abs(gain));
for i = 1:numel(gain)
    if gain(i) >= 0
        text(i, gain(i) + offset, sprintf('%+.2f', gain(i)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 9);
    else
        text(i, gain(i) - offset, sprintf('%+.2f', gain(i)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontSize', 9);
    end
end
ylim([min(gain) - 4*offset, max(gain) + 4*offset]);

xlabel('Scenario');
ylabel(sprintf('Gain %s : statique - adaptatif (%s)', label, unit));
title(sprintf('%s %s : apport de l''adaptation en ligne (positif = adaptatif meilleur)', ...
    label, periodLabel));
subtitle('Exécution de référence, graine unique — non concluant seul pour les écarts de quelques dixièmes de dB');
legend('Location', 'best');
hold off;
saveas(fig, fullfile(cfg.paths.figures, ...
    sprintf('Adaptation_gain_%s_%s.png', upper(metric), lower(period))));
close(fig);
end
