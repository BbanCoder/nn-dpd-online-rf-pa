function net = createRVTDNN(inputSize, cfg)
%CREATERVTDNN Create a real-valued time-delay neural network as dlnetwork.
if inputSize <= 0
    error('inputSize must be positive.');
end
if ~(isfield(cfg,'toolboxes') && cfg.toolboxes.deepLearning)
    net = [];
    return;
end
layers = [featureInputLayer(inputSize,'Normalization','none','Name','features')];
for k = 1:numel(cfg.nn.hiddenUnits)
    layers = [layers; fullyConnectedLayer(cfg.nn.hiddenUnits(k),'Name',sprintf('fc%d',k)); tanhLayer('Name',sprintf('tanh%d',k))]; %#ok<AGROW>
end
layers = [layers; fullyConnectedLayer(2,'Name','fc_out')];
net = dlnetwork(layerGraph(layers));
end
