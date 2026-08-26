function result = scenario_S5_LoadMismatch(baseCfg)
%SCENARIO_S5_LOADMISMATCH Scenario S5: variation de charge, désadaptation ou effet VSWR.
% Objectifs couverts du mémoire :
% 1) étude des mécanismes de variation du HPA ;
% 2) implémentation d une NN-DPD avec adaptation en ligne ;
% 3) définition de scénarios dynamiques ;
% 4) comparaison DPD statique versus DPD adaptative ;
% 5) évaluation par ACPR, NMSE et EVM.
rng(42);
cfg = scenarioConfig(baseCfg, 'S5');
result = runDPDScenario(cfg);
end
