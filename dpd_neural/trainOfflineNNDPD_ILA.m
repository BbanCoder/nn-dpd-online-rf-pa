function model = trainOfflineNNDPD_ILA(paOutput, referenceInput, cfg)
%TRAINOFFLINENNDPD_ILA Train a NN-DPD using indirect learning architecture.
paOutput = paOutput(:); referenceInput = referenceInput(:);
if numel(paOutput) ~= numel(referenceInput)
    error('paOutput and referenceInput must have same length.');
end
[X, T, ~] = makeNNFeatures(paOutput, referenceInput, cfg);
model = struct();
model.type = 'NN_DPD_ILA';
model.memoryDepth = cfg.nn.memoryDepth;
model.outputClip = cfg.nn.outputClip;
model.hasDeepLearning = isfield(cfg,'toolboxes') && cfg.toolboxes.deepLearning;
model.normalization.mu = mean(X,2);
model.normalization.sigma = std(X,0,2) + 1e-8;
Xn = (X - model.normalization.mu) ./ model.normalization.sigma;

if ~model.hasDeepLearning
    % Ridge fallback used only when Deep Learning Toolbox is absent.
    A = [Xn; ones(1,size(Xn,2))].';
    lambda = 1e-4;
    W = (A'*A + lambda*eye(size(A,2))) \ (A'*T.');
    model.fallbackW = W;
    model.net = [];
    model.trainingLoss = mean(sum((A*W - T.').^2,2));
    return;
end

net = createRVTDNN(size(Xn,1), cfg);
avgGrad = [];
avgSqGrad = [];
iter = 0;
N = size(Xn,2);
mb = min(cfg.nn.miniBatchSize, N);
lossHistory = [];
for epoch = 1:cfg.nn.maxEpochsOffline
    order = randperm(N);
    for start = 1:mb:N
        iter = iter + 1;
        ids = order(start:min(start+mb-1,N));
        dlX = dlarray(single(Xn(:,ids)), 'CB');
        dlT = dlarray(single(T(:,ids)), 'CB');
        [loss, gradients] = dlfeval(@localModelLoss, net, dlX, dlT);
        [net, avgGrad, avgSqGrad] = adamupdate(net, gradients, avgGrad, avgSqGrad, iter, cfg.nn.learningRateOffline);
        lossHistory(end+1) = double(gather(extractdata(loss))); %#ok<AGROW>
    end
end
model.net = net;
model.trainingLoss = lossHistory;
end

function [loss, gradients] = localModelLoss(net, dlX, dlT)
dlY = forward(net, dlX);
loss = mean(sum((dlY - dlT).^2,1));
gradients = dlgradient(loss, net.Learnables);
end
