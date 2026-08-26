function plotAdaptationEvents(timeline, scenarioId, cfg)
%PLOTADAPTATIONEVENTS Save adaptation event timeline.
if isempty(timeline.block)
    return;
end
fig = figure('Visible', cfg.plots.visible);
stem(timeline.block, timeline.adaptEvent, 'filled'); grid on;
xlabel('Block index'); ylabel('Adaptation event'); ylim([-0.1 1.1]);
title(sprintf('%s - Online adaptation events', scenarioId));
saveas(fig, fullfile(cfg.paths.figures, sprintf('%s_adaptation_events.png', scenarioId)));
close(fig);
end
