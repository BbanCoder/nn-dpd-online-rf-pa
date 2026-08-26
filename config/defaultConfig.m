function cfg = defaultConfig(projectRoot)
%DEFAULTCONFIG Centralized configuration for NN-DPD online PA simulations.
if nargin < 1 || isempty(projectRoot)
    projectRoot = fileparts(fileparts(mfilename('fullpath')));
end

cfg = struct();
cfg.projectRoot = projectRoot;
cfg.rngSeed = 42;
cfg.use5GToolboxWhenAvailable = true;
cfg.fastValidation = false;

cfg.paths.root = projectRoot;
cfg.paths.figures = fullfile(projectRoot, 'results', 'figures');
cfg.paths.tables = fullfile(projectRoot, 'results', 'tables');
cfg.paths.mat = fullfile(projectRoot, 'results', 'mat');

% Waveform parameters
cfg.fs = 122.88e6;
cfg.bandwidth = 40e6;
cfg.waveform = 'OFDM-QAM';
cfg.modulation = '64QAM';
cfg.qamM = 64;
cfg.nFFT = 1024;
cfg.cpLen = 72;
% Subcarrier spacing is fs/nFFT = 120 kHz; 334 active subcarriers occupy
% 40.1 MHz, consistent with cfg.bandwidth (used as the ACPR useful channel)
% while leaving room for the adjacent channel below Nyquist (61.44 MHz).
% Wideband operation keeps PA memory effects significant, so that the
% simulated drifts genuinely degrade a static DPD.
cfg.nSubcarriers = 334;
cfg.nOFDMSymbols = 80;
cfg.targetSamples = [];

% PA parameters
cfg.pa.model = 'MP';
cfg.pa.orders = [1 3 5 7];
cfg.pa.memoryDepth = 3;
cfg.pa.gmpLagDepth = 2;
cfg.pa.smallSignalGain = 1.0;
cfg.pa.satLevel = 1.4;
cfg.pa.rappSmoothness = 2.5;
cfg.pa.ampm = 0.15;
cfg.pa.noiseStd = 0.0;
cfg.pa.coeffs = defaultPACoeffs(cfg.pa.orders, cfg.pa.memoryDepth);

% DPD classical parameters
cfg.dpd.orders = [1 3 5 7];
cfg.dpd.memoryDepth = 3;
cfg.dpd.gmpLagDepth = 2;
cfg.dpd.regularization = 1e-5;
cfg.dpd.outputClip = 2.5;

% Neural DPD parameters
cfg.nn.enabled = true;
cfg.nn.architecture = 'RVTDNN';
cfg.nn.memoryDepth = 4;
cfg.nn.hiddenUnits = [48 32];
cfg.nn.maxEpochsOffline = 5;
cfg.nn.miniBatchSize = 512;
cfg.nn.learningRateOffline = 5e-4;
cfg.nn.learningRateOnline = 5e-5;
cfg.nn.outputClip = 2.5;

% ACPR measurement geometry. offsetFactor = 0 is the normative ACLR position
% of TS 38.104 (one channel wide, one channel spacing away). It is shifted
% outwards here because the unfiltered OFDM waveform of this bench leaks
% -20.0 dBc at that position, leaving only 5.6 dB of range between the source
% signal and the un-linearized PA output. See computeACPR.m for the measured
% trade-off. The ACPR reported by this bench is a relative indicator of
% spectral regrowth, never a conformity figure.
cfg.acpr.offsetFactor = 0.25;

% Adaptation thresholds
cfg.thresholds.NMSE_dB = -30;
cfg.thresholds.EVM_percent = 3.5;
cfg.thresholds.ACPR_dBc = -45;

% Relative trigger: adapt when NMSE degrades by more than relNMSE_dB with
% respect to the best (healthy) NMSE observed so far. Absolute thresholds
% above remain conformity criteria but are unreachable at this drive level
% and would otherwise trigger continuously, wasting the adaptation budget.
cfg.adaptation.relNMSE_dB = 2;
cfg.adaptation.minSamples = 4096;
cfg.adaptation.blockSize = 4096;
cfg.adaptation.miniBatchSize = 512;
cfg.adaptation.maxEpochsOnline = 3;
cfg.adaptation.learningRateOnline = 5e-5;
cfg.adaptation.replayRatio = 1.0;
% Proximal online updates: L2 penalty pulling the weights toward their
% pre-update values (light EWC-style regularization). Prevents the few
% thousand samples of one block from distorting the network response in
% the rarely-sampled peak region (catastrophic forgetting at high |x|).
cfg.adaptation.proximalMu = 0.3;
cfg.adaptation.freezeBackbone = true;   % Groupe A #2: online updates adapt only the output layer
cfg.adaptation.cooldownBlocks = 1;
cfg.adaptation.maxAdaptations = 10;

% RL policy weights
cfg.rl.weights = [1.0 0.5 0.05 0.1];
cfg.rl.computeBudget = 1.0;

% Plot settings
cfg.plots.visible = 'off';
cfg.plots.savePNG = true;
end

function coeffs = defaultPACoeffs(orders, memoryDepth)
coeffs = zeros(numel(orders), memoryDepth+1);
for i = 1:numel(orders)
    p = orders(i);
    for m = 0:memoryDepth
        base = (0.90)^(m) / p;
        if p == 1
            val = 1.00 * (0.65)^m;
        else
            val = ((-0.18)/(p-1)) * (0.55)^m;
        end
        coeffs(i,m+1) = base * val * exp(1j*(0.04*p + 0.02*m));
    end
end
coeffs(1,1) = 1.0 + 0.02j;
end
