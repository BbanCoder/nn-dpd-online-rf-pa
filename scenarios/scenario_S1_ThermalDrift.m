function result = scenario_S1_ThermalDrift(baseCfg)
%SCENARIO_S1_THERMALDRIFT Scenario S1: dérive thermique graduelle du PA.
% Objectifs couverts du mémoire :
% 1) étude des mécanismes de variation du HPA ;
% 2) implémentation d une NN-DPD avec adaptation en ligne ;
% 3) définition de scénarios dynamiques ;
% 4) comparaison DPD statique versus DPD adaptative ;
% 5) évaluation par ACPR, NMSE et EVM.
rng(42);
cfg = scenarioConfig(baseCfg, 'S1');
result = runDPDScenario(cfg);
end
