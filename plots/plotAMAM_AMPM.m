function plotAMAM_AMPM(x, yNom, yDyn, scenarioId, cfg)
%PLOTAMAM_AMPM Save AM/AM and AM/PM curves for nominal and dynamic PA.
N = min([numel(x), numel(yNom), numel(yDyn), 8000]);
x = x(1:N); yNom = yNom(1:N); yDyn = yDyn(1:N);
fig = figure('Visible', cfg.plots.visible);
subplot(1,2,1); hold on; grid on;
scatter(abs(x), abs(yNom), 6, 'filled', 'MarkerFaceAlpha', 0.5, 'DisplayName','Nominal');
scatter(abs(x), abs(yDyn), 6, 'filled', 'MarkerFaceAlpha', 0.5, 'DisplayName','Dynamic');
xlabel('|x|'); ylabel('|y|'); title('AM/AM'); legend('Location','best');
subplot(1,2,2); hold on; grid on;
% Per-sample AM/PM = phase rotation introduced by the PA, wrapped to (-180,180] deg.
% (No unwrap: samples are not ordered by |x|, so unwrap yielded spurious 2*pi bands.)
scatter(abs(x), rad2deg(angle(yNom./(x+eps))), 6, 'filled', 'MarkerFaceAlpha', 0.5, 'DisplayName','Nominal');
scatter(abs(x), rad2deg(angle(yDyn./(x+eps))), 6, 'filled', 'MarkerFaceAlpha', 0.5, 'DisplayName','Dynamic');
xlabel('|x|'); ylabel('Phase distortion (deg)'); title('AM/PM'); legend('Location','best');
saveas(fig, fullfile(cfg.paths.figures, sprintf('%s_AMAM_AMPM.png', scenarioId)));
close(fig);
end
