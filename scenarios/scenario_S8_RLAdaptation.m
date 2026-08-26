function result = scenario_S8_RLAdaptation(baseCfg)
%SCENARIO_S8_RLADAPTATION Scenario S8: optimisation de la politique d adaptation par Reinforcement Learning ou heuristique.
% Objectifs couverts du mémoire :
% 1) étude des mécanismes de variation du HPA ;
% 2) implémentation d une NN-DPD avec adaptation en ligne ;
% 3) définition de scénarios dynamiques ;
% 4) comparaison DPD statique versus DPD adaptative ;
% 5) évaluation par ACPR, NMSE et EVM.
rng(42);
cfg = scenarioConfig(baseCfg, 'S8');
result = runDPDScenario(cfg);
end
