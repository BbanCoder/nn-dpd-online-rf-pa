function plotMetricTimeline(timeline, scenarioId, cfg)
%PLOTMETRICTIMELINE Save NMSE/EVM/ACPR timelines.
if isempty(timeline.block)
    return;
end
fig = figure('Visible', cfg.plots.visible);
subplot(3,1,1); hold on; grid on; ylabel('NMSE (dB)');
p1 = plot(timeline.block, timeline.NMSE_static, '-o', 'DisplayName','NN static');
p2 = plot(timeline.block, timeline.NMSE_adapt, '-s', 'DisplayName','NN adaptive');
yline(cfg.thresholds.NMSE_dB, '--', 'Seuil'); legend([p1 p2],'Location','best');
subplot(3,1,2); hold on; grid on; ylabel('EVM (%)');
p1 = plot(timeline.block, timeline.EVM_static, '-o', 'DisplayName','NN static');
p2 = plot(timeline.block, timeline.EVM_adapt, '-s', 'DisplayName','NN adaptive');
yline(cfg.thresholds.EVM_percent, '--', 'Seuil'); legend([p1 p2],'Location','best');
subplot(3,1,3); hold on; grid on; ylabel('ACPR (dBc)'); xlabel('Block index');
p1 = plot(timeline.block, timeline.ACPR_static, '-o', 'DisplayName','NN static');
p2 = plot(timeline.block, timeline.ACPR_adapt, '-s', 'DisplayName','NN adaptive');
yline(cfg.thresholds.ACPR_dBc, '--', 'Seuil'); legend([p1 p2],'Location','best');
sgtitle(sprintf('%s - Metric timelines', scenarioId));
saveas(fig, fullfile(cfg.paths.figures, sprintf('%s_metric_timeline.png', scenarioId)));
close(fig);
end
