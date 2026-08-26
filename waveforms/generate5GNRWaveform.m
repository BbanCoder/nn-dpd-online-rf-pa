function [x, info] = generate5GNRWaveform(cfg)
%GENERATE5GNRWAVEFORM Generate a 5G NR downlink waveform from the project configuration.
%
%   The carrier is built from cfg, NOT from the toolbox defaults: subcarrier
%   spacing is derived from cfg.fs/cfg.nFFT, the grid size from
%   cfg.nSubcarriers, the channel bandwidth from cfg.bandwidth and the PDSCH
%   modulation from cfg.modulation. An earlier version called
%   nrDLCarrierConfig with no arguments, which silently produced the toolbox
%   default carrier (52 RB, 15 kHz SCS, QPSK, 15.36 MHz) whatever the
%   configuration asked for, while info still reported the requested values.
%
%   Only the PDSCH is generated: SS burst and PDCCH are disabled. For a PA
%   linearization study the payload carries the signal statistics that matter
%   (bandwidth, numerology, modulation, PAPR); the control channels would add
%   protocol structure without changing the linearization problem.
%
%   The toolbox picks its own sample rate from SCS and grid size (61.44 MHz
%   for 32 RB at 120 kHz). The waveform is resampled to cfg.fs so that every
%   scenario of the bench shares one sample rate, and so that the ACPR
%   adjacent window has spectrum to sit in.
%
%   If generation fails, the function falls back to the internal OFDM/QAM
%   generator and says so LOUDLY in info.fallbackReason: the previous silent
%   fallback is how the configuration mismatch went unnoticed.
arguments
    cfg struct
end
x = [];
info = struct();
info.fallbackReason = '';

wanted5G = isfield(cfg,'toolboxes') && cfg.toolboxes.fiveG && cfg.use5GToolboxWhenAvailable;
if ~wanted5G
    [x, info] = generateQAMOFDMWaveform(cfg);
    info.waveform = 'OFDM-QAM (toolbox 5G indisponible ou desactive)';
    info.fallbackReason = '5G Toolbox unavailable or disabled by cfg.use5GToolboxWhenAvailable';
    return;
end

try
    scsKHz = round(cfg.fs / cfg.nFFT / 1e3);          % 122.88e6/1024 = 120 kHz
    valid = [15 30 60 120];
    if ~ismember(scsKHz, valid)
        [~, k] = min(abs(valid - scsKHz));
        scsKHz = valid(k);
    end
    fr = 'FR1';
    if scsKHz >= 120, fr = 'FR2'; end

    nrbWanted = max(1, floor(cfg.nSubcarriers / 12));
    bwMHz = round(cfg.bandwidth / 1e6);

    % The maximum grid size depends on bandwidth and SCS (TS 38.104). Rather
    % than duplicate that table, shrink until the toolbox accepts.
    lastErr = '';
    w = [];
    for nrb = nrbWanted:-1:1
        try
            d = buildCarrierLocal(fr, bwMHz, scsKHz, nrb, cfg.modulation);
            w = nrWaveformGenerator(d);
            break;
        catch ME
            lastErr = ME.message;
        end
    end
    if isempty(w)
        error('nrWaveformGenerator rejected every grid size (last: %s)', lastErr);
    end

    fsNative = size(w,1) / (d.NumSubframes * 1e-3);
    x = w(:,1);

    % Resample to the bench sample rate (band-limited, integer ratio here).
    if abs(fsNative - cfg.fs) > 1
        x = resampleToLocal(x, fsNative, cfg.fs);
    end
    x = normalizeSignal(x);

    info.waveform        = '5GNR';
    info.modulation      = cfg.modulation;
    info.fs              = cfg.fs;
    info.fsNative        = fsNative;
    info.frequencyRange  = fr;
    info.subcarrierSpacing_kHz = scsKHz;
    info.nSizeGrid_RB    = nrb;
    info.nSubcarriers    = nrb*12;
    info.occupiedBW_MHz  = nrb*12*scsKHz*1e3/1e6;
    info.bandwidth       = cfg.bandwidth;
    info.nOFDMSymbols    = NaN;
catch ME
    warning('generate5GNRWaveform:fallback', ...
        ['Generation 5G NR echouee, repli sur OFDM/QAM interne. ' ...
         'Les resultats du scenario NE SONT PAS ceux d''une waveform 5G NR. Raison : %s'], ME.message);
    [x, info] = generateQAMOFDMWaveform(cfg);
    info.waveform = 'OFDM-QAM (repli apres echec 5G NR)';
    info.fallbackReason = ME.message;
end
end

% -------------------------------------------------------------------------
function d = buildCarrierLocal(fr, bwMHz, scsKHz, nrb, modulation)
d = nrDLCarrierConfig;
d.FrequencyRange   = fr;
d.ChannelBandwidth = bwMHz;
% One subframe (1 ms) gives 122 880 samples at 122.88 MHz, i.e. 30 adaptation
% blocks -- comparable to the 22 blocks of the OFDM/QAM scenarios. Ten
% subframes would give 300 blocks and change the scenario dynamics entirely.
d.NumSubframes     = 1;

d.SCSCarriers{1}.SubcarrierSpacing = scsKHz;
d.SCSCarriers{1}.NSizeGrid  = nrb;
d.SCSCarriers{1}.NStartGrid = 0;

d.BandwidthParts{1}.SubcarrierSpacing = scsKHz;
d.BandwidthParts{1}.NSizeBWP  = nrb;
d.BandwidthParts{1}.NStartBWP = 0;

d.PDSCH{1}.Enable     = true;
d.PDSCH{1}.Modulation = modulation;
d.PDSCH{1}.PRBSet     = 0:nrb-1;

% Payload only: control channels would not change the linearization problem.
d.SSBurst.Enable = false;
for k = 1:numel(d.PDCCH),   d.PDCCH{k}.Enable = false; end
for k = 1:numel(d.CORESET)
    d.CORESET{k}.FrequencyResources = ones(1, max(1, floor(nrb/6)));
end
end

% -------------------------------------------------------------------------
function y = resampleToLocal(x, fsIn, fsOut)
%RESAMPLETOLOCAL Band-limited resampling, using resample() when available and
%   an FFT zero-padding interpolation otherwise (exact for integer ratios on
%   a band-limited signal, which is the case here).
if exist('resample', 'file') == 2
    [p, q] = rat(fsOut/fsIn, 1e-9);
    y = resample(x, p, q);
    return;
end
r = fsOut/fsIn;
N = numel(x);
M = round(N*r);
X = fftshift(fft(x));
if M > N
    pad = (M - N)/2;
    Y = [zeros(floor(pad),1); X; zeros(ceil(pad),1)];
else
    cut = (N - M)/2;
    Y = X(floor(cut)+1 : floor(cut)+M);
end
y = ifft(ifftshift(Y)) * (M/N);
end
