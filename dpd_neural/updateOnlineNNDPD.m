function [model, info, replayBuffer] = updateOnlineNNDPD(model, paOutputNew, referenceInputNew, cfg, replayBuffer)
%UPDATEONLINENNDPD Fine-tune NN-DPD using recent PA output and replay samples.
if nargin < 5 || isempty(replayBuffer)
    replayBuffer = struct('input',[],'target',[]);
end
tStart = tic;
paOutputNew = paOutputNew(:); referenceInputNew = referenceInputNew(:);
if numel(paOutputNew) ~= numel(referenceInputNew)
    error('paOutputNew and referenceInputNew must have same length.');
end
[paTrain, refTrain, replayBuffer] = replayBufferUpdate(replayBuffer, paOutputNew, referenceInputNew, cfg);
[X, T, ~] = makeNNFeatures(paTrain, refTrain, cfg);
Xn = (X - model.normalization.mu) ./ model.normalization.sigma;
oldModel = model;

% Held-out validation split (25%): the divergence guard compares the loss
% on samples NOT used for the update, so that overfitting the recent block
% (which lowers the training loss while degrading generalization) is
% detected and reverted.
Ntot = size(Xn,2);
perm = randperm(Ntot);
nVal = max(1, round(0.25*Ntot));
valIds = perm(1:nVal);
trnIds = perm(nVal+1:end);
if isempty(trnIds), trnIds = valIds; end
XnT = Xn(:,trnIds); TT = T(:,trnIds);
XnV = Xn(:,valIds); TV = T(:,valIds);
% --- Garde-fou v2 : la perte est mesuree sur le jeu d'ANCRAGE, fixe depuis
% l'entrainement hors ligne et jamais employe pour une mise a jour. Le
% jugement sur les donnees du bloc (XnV) ne detecterait pas une perte de
% generalisation, puisque s'ajuster au bloc fait baisser cette perte-la.
useAnchor = isfield(model,'anchor') && ~isempty(model.anchor);
if useAnchor
    XnV = model.anchor.Xn; TV = model.anchor.T;
end
oldLoss = evaluateLossLocal(model, XnV, TV);
% --- Garde-fou de bout en bout : NMSE reel apres predistorsion + PA nominal.
% L'erreur quadratique sur le post-inverse ne voit pas une degradation
% concentree dans les cretes, la ou le PA sature ; le NMSE la voit.
e2e = useAnchor && isfield(model.anchor,'xRef') && isfield(cfg,'pa');
if e2e
    oldNMSE = anchorNMSELocal(model, cfg);
end

if ~(isfield(model,'hasDeepLearning') && model.hasDeepLearning && ~isempty(model.net))
    A = [XnT; ones(1,size(XnT,2))].';
    lambda = 1e-4;
    model.fallbackW = (A'*A + lambda*eye(size(A,2))) \ (A'*TT.');
    newLoss = evaluateLossLocal(model, XnV, TV);
    info = struct('lossBefore',oldLoss,'lossAfter',newLoss,'iterations',1,'runtime_s',toc(tStart),'reverted',false);
    return;
end

net = model.net;
refVals = net.Learnables.Value;   % anchor for the proximal penalty
mu = 0;
if isfield(cfg.adaptation, 'proximalMu'), mu = cfg.adaptation.proximalMu; end
% Groupe A #2: frozen backbone -- adapt only the output layer 'fc_out' online.
% Far fewer parameters => data-efficient, low-variance updates on small blocks.
freezeBackbone = isfield(cfg.adaptation,'freezeBackbone') && cfg.adaptation.freezeBackbone;
avgGrad = [];
avgSqGrad = [];
iter = 0;
N = size(XnT,2);
mb = min(cfg.adaptation.miniBatchSize, N);
for epoch = 1:cfg.adaptation.maxEpochsOnline
    order = randperm(N);
    for start = 1:mb:N
        iter = iter + 1;
        ids = order(start:min(start+mb-1,N));
        dlX = dlarray(single(XnT(:,ids)), 'CB');
        dlT = dlarray(single(TT(:,ids)), 'CB');
        [loss, gradients] = dlfeval(@localModelLoss, net, dlX, dlT, refVals, mu); %#ok<ASGLU>
        if freezeBackbone
            for kk = 1:height(gradients)
                if ~strcmp(string(gradients.Layer(kk)), "fc_out")
                    gradients.Value{kk} = 0 * gradients.Value{kk};   % freeze this layer
                end
            end
        end
        [net, avgGrad, avgSqGrad] = adamupdate(net, gradients, avgGrad, avgSqGrad, iter, cfg.adaptation.learningRateOnline);
    end
end
model.net = net;
newLoss = evaluateLossLocal(model, XnV, TV);
reverted = false;
tol = 1.05;
if isfield(cfg.adaptation,'anchorTolerance'), tol = cfg.adaptation.anchorTolerance; end
rejet = newLoss > tol*max(oldLoss, eps);
if e2e
    newNMSE = anchorNMSELocal(model, cfg);
    tolDB = 0.10; if isfield(cfg.adaptation,'anchorToleranceDB'), tolDB = cfg.adaptation.anchorToleranceDB; end
    rejet = rejet || (newNMSE > oldNMSE + tolDB);
end
if rejet
    model = oldModel;
    newLoss = oldLoss;
    reverted = true;
end
info = struct('lossBefore',oldLoss,'lossAfter',newLoss,'iterations',iter,'runtime_s',toc(tStart),'reverted',reverted);
end

function loss = evaluateLossLocal(model, Xn, T)
if isfield(model,'hasDeepLearning') && model.hasDeepLearning && ~isempty(model.net)
    dlY = forward(model.net, dlarray(single(Xn),'CB'));
    Y = double(gather(extractdata(dlY)));
else
    A = [Xn; ones(1,size(Xn,2))].';
    Y = (A * model.fallbackW).';
end
loss = mean(sum((Y - T).^2,1));
end

function [loss, gradients] = localModelLoss(net, dlX, dlT, refVals, mu)
dlY = forward(net, dlX);
loss = mean(sum((dlY - dlT).^2,1));
if nargin >= 5 && mu > 0
    vals = net.Learnables.Value;
    pen = 0;
    for k = 1:numel(vals)
        pen = pen + sum((vals{k} - refVals{k}).^2, 'all');
    end
    loss = loss + mu * pen;
end
gradients = dlgradient(loss, net.Learnables);
end

function nm = anchorNMSELocal(model, cfg)
x = model.anchor.xRef(:);
z = applyNNDPD(x, model, cfg);
prof = generatePADriftProfile(numel(x), cfg, 'nominal');
y = paDynamicModel(z, cfg, prof, upper(cfg.pa.model));
nm = computeNMSE(y, x);
end
