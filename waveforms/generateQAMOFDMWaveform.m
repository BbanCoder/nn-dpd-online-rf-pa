function [x, info] = generateQAMOFDMWaveform(cfg)
%GENERATEQAMOFDMWAVEFORM Generate an internal OFDM/QAM waveform without external data.
arguments
    cfg struct
end
rng(cfg.rngSeed);
Nfft = cfg.nFFT;
Nsc = min(cfg.nSubcarriers, Nfft-16);
Nsym = cfg.nOFDMSymbols;
cpLen = cfg.cpLen;
M = cfg.qamM;

if Nsc <= 0 || mod(Nsc,2) ~= 0
    error('nSubcarriers must be positive and even.');
end

numQam = Nsc * Nsym;
data = randi([0 M-1], numQam, 1);
qam = localQAMMod(data, M);
grid = reshape(qam, Nsc, Nsym);

freqGrid = zeros(Nfft, Nsym);
startIdx = floor((Nfft - Nsc)/2) + 1;
freqGrid(startIdx:startIdx+Nsc-1, :) = grid;
freqGrid = ifftshift(freqGrid, 1);

timeGrid = ifft(freqGrid, Nfft, 1) * sqrt(Nfft);
withCP = [timeGrid(end-cpLen+1:end,:); timeGrid];
x = withCP(:);
x = normalizeSignal(x);

info = struct();
info.waveform = 'OFDM-QAM';
info.modulation = sprintf('%dQAM', M);
info.qamM = M;
info.fs = cfg.fs;
info.bandwidth = cfg.bandwidth;
info.nFFT = Nfft;
info.cpLen = cpLen;
info.nSubcarriers = Nsc;
info.nOFDMSymbols = Nsym;
info.data = data;
info.symbols = qam;
end

function s = localQAMMod(data, M)
% Square QAM with average power normalization. Does not require Communications Toolbox.
L = sqrt(M);
if abs(L - round(L)) > eps
    error('Only square QAM constellations are supported.');
end
L = round(L);
I = mod(data, L);
Q = floor(data / L);
levels = -(L-1):2:(L-1);
s = levels(I+1).' + 1j*levels(Q+1).';
s = s(:);
s = s / sqrt(mean(abs(s).^2));
end
