function result = scenario_S0_NominalCalibration(baseCfg)
%SCENARIO_S0_NOMINALCALIBRATION Scenario S0: calibration nominale et référence de performance.
% Objectifs couverts du mémoire :
% 1) étude des mécanismes de variation du HPA ;
% 2) implémentation d une NN-DPD avec adaptation en ligne ;
% 3) définition de scénarios dynamiques ;
% 4) comparaison DPD statique versus DPD adaptative ;
% 5) évaluation par ACPR, NMSE et EVM.
rng(42);
cfg = scenarioConfig(baseCfg, 'S0');
result = runDPDScenario(cfg);
end
