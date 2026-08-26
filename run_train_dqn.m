function agent = run_train_dqn(numEpisodes)
%RUN_TRAIN_DQN Train a real DQN agent for the DPD adaptation policy (S8).
%
%   Requires the Reinforcement Learning Toolbox + Deep Learning Toolbox
%   (MATLAB R2023a). Training is long (the environment replays the DPD
%   chain at every step): budget for hours at ~1000 episodes.
%
%   run_train_dqn            % 1000 episodes (default)
%   run_train_dqn(100)       % shorter run for a first smoke test
%
%   The reward uses cfg.rl.weights = [w1 w2 w3 w4] (EVM/NMSE/ACPR badness
%   and action cost). Saves the agent to rl_policy/dqn_agent.mat; any S8
%   run afterwards loads it automatically (trainDQNAdaptationAgent).
%   Without this file S8 keeps using the heuristic policy.

if nargin < 1 || isempty(numEpisodes), numEpisodes = 1000; end

projectRoot = fileparts(mfilename('fullpath'));
addpath(genpath(projectRoot));
cfg = defaultConfig(projectRoot);
cfg.toolboxes = toolboxCheck();
assert(cfg.toolboxes.reinforcementLearning, 'Reinforcement Learning Toolbox requise.');
assert(cfg.toolboxes.deepLearning, 'Deep Learning Toolbox requise.');

% --- Reference waveform + nominal PA + offline NN-DPD (trained ONCE).
rng(cfg.rngSeed);
[x, ~] = generateQAMOFDMWaveform(cfg);
x = normalizeSignal(x);
nominalProfile = generatePADriftProfile(numel(x), cfg, 'nominal');
yNom = paDynamicModel(x, cfg, nominalProfile, cfg.pa.model);
fprintf('Entraînement hors ligne du NN-DPD de base...\n');
baseModel = trainOfflineNNDPD_ILA(yNom, x, cfg);

% Crest anchor for the peak-window replay store (same as the engine).
[~, crestIdx] = max(abs(x));
lo = max(1, crestIdx-64); hi = min(numel(x), crestIdx+63);
cfg.rl.crestSeed = {yNom(lo:hi), x(lo:hi), max(abs(x))};

% --- Environment.
env = createDPDAdaptationEnv(cfg, baseModel, x);
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

% --- Q-network : obs(6) -> Q-value per action (6).
net = [ featureInputLayer(obsInfo.Dimension(1), 'Name', 'state')
        fullyConnectedLayer(64, 'Name', 'fc1'); reluLayer('Name', 'relu1')
        fullyConnectedLayer(64, 'Name', 'fc2'); reluLayer('Name', 'relu2')
        fullyConnectedLayer(numel(actInfo.Elements), 'Name', 'qvalues') ];
critic = rlVectorQValueFunction(net, obsInfo, actInfo);

% --- DQN agent.
agentOpts = rlDQNAgentOptions( ...
    'UseDoubleDQN', true, ...
    'SampleTime', 1, ...
    'DiscountFactor', 0.95, ...
    'MiniBatchSize', 64, ...
    'ExperienceBufferLength', 1e5, ...
    'TargetSmoothFactor', 1e-3);
agentOpts.EpsilonGreedyExploration.Epsilon = 1.0;
agentOpts.EpsilonGreedyExploration.EpsilonMin = 0.05;
% Decay per agent step (22 steps/episode): 3e-4 keeps exploration alive over
% most of a 1000-episode run; 3e-3 collapsed it within ~45 episodes and the
% average reward degraded after its early peak.
agentOpts.EpsilonGreedyExploration.EpsilonDecay = 3e-4;
agentOpts.CriticOptimizerOptions.LearnRate = 1e-3;
agent = rlDQNAgent(critic, agentOpts);

% --- Training.
maxSteps = ceil(numel(x)/cfg.adaptation.blockSize) + 1;
trainOpts = rlTrainingOptions( ...
    'MaxEpisodes', numEpisodes, ...
    'MaxStepsPerEpisode', maxSteps, ...
    'ScoreAveragingWindowLength', 20, ...
    'Verbose', true, ...
    'Plots', 'none', ...
    'StopTrainingCriteria', 'EpisodeCount', ...
    'StopTrainingValue', numEpisodes);

fprintf('Entraînement DQN (%d épisodes, %d pas max)...\n', numEpisodes, maxSteps);
train(agent, env, trainOpts);

% --- Save.
save(fullfile(projectRoot, 'rl_policy', 'dqn_agent.mat'), 'agent', '-v7.3');
fprintf('\nAgent DQN sauvegardé dans rl_policy/dqn_agent.mat.\nRelancez run_all : le scénario S8 utilisera automatiquement cet agent.\n');
end
