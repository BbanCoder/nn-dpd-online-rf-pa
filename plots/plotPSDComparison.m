function plotPSDComparison(signals, fs, scenarioId, cfg)
%PLOTPSDCOMPARISON Save comparative PSD figure.
fig = figure('Visible', cfg.plots.visible);
hold on; grid on;
names = fieldnames(signals);
for i = 1:numel(names)
    x = signals.(names{i})(:);
    [f, Pdb] = localPSD(x, fs);
    plot(f/1e6, Pdb, 'DisplayName', strrep(names{i},'_','\_'));
end
xlabel('Frequency (MHz)'); ylabel('Normalized PSD (dB)');
title(sprintf('%s - PSD comparison', scenarioId)); legend('Location','best');
saveas(fig, fullfile(cfg.paths.figures, sprintf('%s_PSD_comparison.png', scenarioId)));
close(fig);
end

function [f, Pdb] = localPSD(x, fs)
Nfft = 2^nextpow2(max(4096, min(numel(x), 2^16)));
seg = x(1:min(numel(x),Nfft));
if numel(seg) < Nfft, seg(end+1:Nfft) = 0; end
w = 0.5 - 0.5*cos(2*pi*(0:Nfft-1)'/(Nfft-1));
X = fftshift(fft(seg(:).*w, Nfft));
P = abs(X).^2;
Pdb = 10*log10(P/max(P)+eps);
f = (-Nfft/2:Nfft/2-1).' * fs/Nfft;
end
