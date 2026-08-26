function cfg = scenarioConfig(baseCfg, scenarioId)
%SCENARIOCONFIG Return a scenario-specific configuration.
validateattributes(scenarioId, {'char','string'}, {'nonempty'});
cfg = baseCfg;
scenarioId = char(scenarioId);
cfg.scenario.id = scenarioId;
cfg.scenario.useRL = false;

switch upper(scenarioId)
    case 'S0'
        cfg.scenario.name = 'Nominal calibration';
        cfg.scenario.driftType = 'nominal';
        cfg.scenario.comment = 'Calibration nominale et référence de performance';
    case 'S1'
        cfg.scenario.name = 'Thermal drift';
        cfg.scenario.driftType = 'thermal';
        cfg.scenario.comment = 'Dérive thermique graduelle du PA';
    case 'S2'
        cfg.scenario.name = 'Power jump';
        cfg.scenario.driftType = 'powerjump';
        cfg.scenario.comment = 'Rupture brutale de puissance ou changement d IBO';
    case 'S3'
        cfg.scenario.name = 'Memory variation';
        cfg.scenario.driftType = 'memory';
        cfg.scenario.comment = 'Variation des effets mémoire du PA';
    case 'S4'
        cfg.scenario.name = 'Aging drift';
        cfg.scenario.driftType = 'aging';
        cfg.scenario.comment = 'Vieillissement lent du PA';
    case 'S5'
        cfg.scenario.name = 'Load mismatch';
        cfg.scenario.driftType = 'load';
        cfg.scenario.comment = 'Variation de charge, désadaptation ou effet VSWR';
    case 'S6'
        cfg.scenario.name = 'Combined dynamic drift';
        cfg.scenario.driftType = 'combined';
        cfg.scenario.comment = 'Dérive combinée thermique + puissance + mémoire + charge';
    case 'S7'
        cfg.scenario.name = '5G NR stress test';
        cfg.scenario.driftType = 'combined';
        cfg.scenario.comment = 'Stress test 5G NR avec variation de bande, modulation et numérologie';
        cfg.waveform = '5GNR';
        cfg.bandwidth = 50e6;
        cfg.qamM = 256;
        cfg.modulation = '256QAM';
        % TS 38.104 caps a 50 MHz FR2 carrier at 32 RB for 120 kHz SCS,
        % i.e. 384 subcarriers = 46.08 MHz occupied. The generator enforces
        % this limit; asking for more is silently reduced.
        cfg.nSubcarriers = 384;
        cfg.nOFDMSymbols = 96;
        cfg.thresholds.EVM_percent = 3.5;
    case 'S8'
        cfg.scenario.name = 'RL or heuristic adaptation policy';
        cfg.scenario.driftType = 'combined';
        cfg.scenario.comment = 'Optimisation de la politique d adaptation par RL ou heuristique';
        cfg.scenario.useRL = true;
    otherwise
        error('Unknown scenario identifier: %s', scenarioId);
end

if isfield(baseCfg,'fastValidation') && baseCfg.fastValidation
    cfg.fastValidation = true;
    cfg.nOFDMSymbols = min(cfg.nOFDMSymbols, 16);
    cfg.adaptation.blockSize = 2048;
    cfg.adaptation.minSamples = 1024;
    cfg.nn.maxEpochsOffline = 1;
    cfg.adaptation.maxEpochsOnline = 1;
end
end
