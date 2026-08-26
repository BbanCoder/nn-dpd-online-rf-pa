function result = scenario_S4_AgingDrift(baseCfg)
%SCENARIO_S4_AGINGDRIFT Scenario S4: vieillissement lent du PA.
% Objectifs couverts du mémoire :
% 1) étude des mécanismes de variation du HPA ;
% 2) implémentation d une NN-DPD avec adaptation en ligne ;
% 3) définition de scénarios dynamiques ;
% 4) comparaison DPD statique versus DPD adaptative ;
% 5) évaluation par ACPR, NMSE et EVM.
rng(42);
cfg = scenarioConfig(baseCfg, 'S4');
result = runDPDScenario(cfg);
end
