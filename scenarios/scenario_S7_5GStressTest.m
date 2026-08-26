function result = scenario_S7_5GStressTest(baseCfg)
%SCENARIO_S7_5GSTRESSTEST Scenario S7: stress test 5G NR avec variation de bande, modulation et numérologie.
% Objectifs couverts du mémoire :
% 1) étude des mécanismes de variation du HPA ;
% 2) implémentation d une NN-DPD avec adaptation en ligne ;
% 3) définition de scénarios dynamiques ;
% 4) comparaison DPD statique versus DPD adaptative ;
% 5) évaluation par ACPR, NMSE et EVM.
rng(42);
cfg = scenarioConfig(baseCfg, 'S7');
result = runDPDScenario(cfg);
end
