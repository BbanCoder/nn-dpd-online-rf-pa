function plotMonteCarloDelta(rawTable, cfg)
%PLOTMONTECARLODELTA Paired per-seed NMSE delta, adaptive minus static.
%   One group per Monte-Carlo scenario, two bars per group (during and
%   after the drift). Error bars are the 95 % confidence interval of the
%   paired difference (Student, n-1 degrees of freedom); a bar whose
%   interval excludes zero is marked with an asterisk.
%
%   Sign convention follows the metric: NMSE improves when it decreases,
%   so a NEGATIVE delta means the adaptive arm did better. This matches
%   the convention used throughout the results chapter.
%
%   The adaptive arm is NN_DPD_Adaptive where it exists and
%   RL_Adaptive_Policy otherwise (scenario S8).
if isempty(rawTable) || ~ismember('Seed', rawTable.Properties.VariableNames)
    return;
end
periods = {'during', 'after'};
periodLabels = {'Pendant la dérive', 'Après la dérive'};

scenarios = sort(unique(string(rawTable.Scenario)));   % S2, S6, S7, S8
nS = numel(scenarios);
delta = nan(nS, numel(periods));
halfCI = nan(nS, numel(periods));
armName = strings(nS, 1);

for i = 1:nS
    rows = rawTable(string(rawTable.Scenario) == scenarios(i), :);
    methods = unique(string(rows.Method));
    if any(methods == "NN_DPD_Adaptive")
        armName(i) = "NN_DPD_Adaptive";
    elseif any(methods == "RL_Adaptive_Policy")
        armName(i) = "RL_Adaptive_Policy";
    else
        continue;
    end
    for p = 1:numel(periods)
        col = sprintf('NMSE_dB_%s', periods{p});
        if ~ismember(col, rows.Properties.VariableNames), continue; end
        st = rows(string(rows.Method) == "NN_DPD_Static", :);
        ad = rows(string(rows.Method) == armName(i), :);
        [~, is, ia] = intersect(st.Seed, ad.Seed);   % pair seed by seed
        d = ad{ia, col} - st{is, col};
        d = d(isfinite(d));
        n = numel(d);
        if n < 2, continue; end
        delta(i, p) = mean(d);
        % Student two-sided 95 % on n-1 dof; falls back to the normal
        % quantile if the Statistics Toolbox is absent.
        if exist('tinv', 'file') == 2
            tcrit = tinv(0.975, n - 1);
        else
            tcrit = 1.96;
        end
        halfCI(i, p) = tcrit * std(d) / sqrt(n);
    end
end

keep = any(isfinite(delta), 2);
if ~any(keep), return; end
scenarios = scenarios(keep); delta = delta(keep, :); halfCI = halfCI(keep, :);
nS = numel(scenarios);

fig = figure('Visible', cfg.plots.visible, 'Position', [100 100 950 500]);
hold on; grid on;
b = bar(categorical(scenarios, scenarios), delta, 'grouped');
for p = 1:numel(periods)
    b(p).DisplayName = periodLabels{p};
end
% Error bars on the true centre of each grouped bar.
for p = 1:numel(periods)
    x = b(p).XEndPoints;
    errorbar(x, delta(:, p), halfCI(:, p), 'k', 'LineStyle', 'none', ...
        'LineWidth', 1, 'CapSize', 8, 'HandleVisibility', 'off');
    for i = 1:nS
        if ~isfinite(delta(i, p)), continue; end
        significant = abs(delta(i, p)) > halfCI(i, p);   % interval excludes zero
        if significant
            yTip = delta(i, p) - sign(delta(i, p)) * 0;
            offset = 1.35 * halfCI(i, p) * sign(delta(i, p));
            text(x(i), yTip + offset, '*', 'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle', 'FontSize', 16, 'FontWeight', 'bold');
        end
    end
end
yline(0, 'k-', 'LineWidth', 1, 'HandleVisibility', 'off');

ylabel('\Delta NMSE apparié : adaptative - statique (dB)');
xlabel('Scenario');
title('Écart apparié par graine entre NN-DPD adaptative et statique (n = 10)');
subtitle('Négatif = adaptative meilleure. Barres : IC 95 % du test apparié ; * = intervalle excluant zéro');
legend('Location', 'best');
hold off;
saveas(fig, fullfile(cfg.paths.figures, 'Synthese_delta_montecarlo.png'));
close(fig);
end
