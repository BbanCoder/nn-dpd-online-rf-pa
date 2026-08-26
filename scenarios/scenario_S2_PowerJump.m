function result = scenario_S2_PowerJump(baseCfg)
%SCENARIO_S2_POWERJUMP Scenario S2: rupture brutale de puissance ou changement d IBO.
% Objectifs couverts du mémoire :
% 1) étude des mécanismes de variation du HPA ;
% 2) implémentation d une NN-DPD avec adaptation en ligne ;
% 3) définition de scénarios dynamiques ;
% 4) comparaison DPD statique versus DPD adaptative ;
% 5) évaluation par ACPR, NMSE et EVM.
rng(42);
cfg = scenarioConfig(baseCfg, 'S2');
result = runDPDScenario(cfg);
end
