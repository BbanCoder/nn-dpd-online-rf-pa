function plotConstellation(signals, scenarioId, cfg, info)
%PLOTCONSTELLATION Save the demodulated QAM constellation for each method.
% For the OFDM-QAM path the PA output of every method is OFDM-demodulated
% (CP removal -> FFT -> active subcarriers) and scalar gain/phase aligned to
% the transmitted reference symbols, so the residual scatter around the ideal
% 64-QAM points is directly readable. Falls back to a time-domain sample cloud
% for the 5G NR path (no internal reference grid available here).
if nargin < 4, info = struct(); end

names = fieldnames(signals);
isOFDM = ~strcmpi(cfg.waveform,'5GNR') && isfield(info,'symbols') ...
    && all(isfield(cfg, {'nFFT','cpLen','nSubcarriers'}));

fig = figure('Visible', cfg.plots.visible);

if isOFDM
    Nfft = cfg.nFFT; cpLen = cfg.cpLen; Nsc = cfg.nSubcarriers;
    symLen = Nfft + cpLen;
    startIdx = floor((Nfft - Nsc)/2) + 1;
    ref = info.symbols(:);
    ideal = unique(ref);                         % 64 ideal constellation points
    lim = 1.35 * max(abs(ideal));
    nM = numel(names); nCol = 3; nRow = ceil(nM/nCol);
    for i = 1:nM
        y = signals.(names{i})(:);
        nSym = floor(numel(y)/symLen);
        Y = reshape(y(1:nSym*symLen), symLen, nSym);
        Y = Y(cpLen+1:end, :);                    % remove cyclic prefix
        F = fftshift(fft(Y, Nfft, 1)/sqrt(Nfft), 1);
        A = F(startIdx:startIdx+Nsc-1, :);        % active subcarriers
        rx = A(:);
        r  = ref(1:min(numel(ref), numel(rx)));
        rx = rx(1:numel(r));
        g  = (rx' * r) / (rx' * rx + eps);        % LS gain/phase alignment
        eq = g * rx;
        evm = 100 * sqrt(mean(abs(eq - r).^2) / mean(abs(r).^2));
        ids = round(linspace(1, numel(eq), min(2500, numel(eq))));
        subplot(nRow, nCol, i); hold on; grid on; axis equal;
        scatter(real(eq(ids)), imag(eq(ids)), 4, 'filled', 'MarkerFaceAlpha', 0.25);
        plot(real(ideal), imag(ideal), 'k+', 'MarkerSize', 5, 'LineWidth', 0.8);
        xlim([-lim lim]); ylim([-lim lim]);
        title(strrep(names{i},'_','\_'), 'FontSize', 9);
        text(-0.92*lim, 0.80*lim, sprintf('EVM %.1f%%', evm), 'FontSize', 8, ...
            'BackgroundColor', 'w', 'Margin', 0.5, 'EdgeColor', [0.7 0.7 0.7]);
        xlabel('I'); ylabel('Q');
    end
    sgtitle(sprintf('%s - Constellation demodulee (64-QAM, alignee en gain)', scenarioId));
else
    hold on; grid on; axis equal;
    for i = 1:numel(names)
        x = signals.(names{i})(:);
        ids = round(linspace(1, numel(x), min(3000, numel(x))));
        scatter(real(x(ids)), imag(x(ids)), 5, 'filled', ...
            'DisplayName', strrep(names{i},'_','\_'));
    end
    xlabel('In-phase'); ylabel('Quadrature');
    title(sprintf('%s - Constellation samples (time-domain)', scenarioId));
    legend('Location','best');
end

saveas(fig, fullfile(cfg.paths.figures, sprintf('%s_constellation.png', scenarioId)));
close(fig);
end
