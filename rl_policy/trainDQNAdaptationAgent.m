function agent = trainDQNAdaptationAgent(cfg)
%TRAINDQNADAPTATIONAGENT Load a trained DQN agent if available, else heuristic.
%   Training itself is done offline by the script run_train_dqn.m, which
%   builds the rlFunctionEnv (createDPDAdaptationEnv), trains a DQN with the
%   reward weighted by cfg.rl.weights, and saves rl_policy/dqn_agent.mat.
%   This function only LOADS that agent so that run_all stays fast; without
%   a saved agent (or without the RL Toolbox) S8 uses the heuristic policy.
agentFile = fullfile(cfg.projectRoot, 'rl_policy', 'dqn_agent.mat');
if exist(agentFile, 'file') == 2
    try
        S = load(agentFile, 'agent');
        agent = S.agent;
        return;
    catch ME
        warning('Could not load %s (%s). Falling back to heuristic policy.', agentFile, ME.message);
    end
end
if ~(isfield(cfg,'toolboxes') && cfg.toolboxes.reinforcementLearning)
    agent = struct('type','heuristic','message','RL Toolbox unavailable; heuristic policy is used.');
else
    agent = struct('type','heuristic','message','No trained agent found. Run run_train_dqn to train and save one.');
end
end
