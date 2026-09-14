function model = trainOfflineNNDPD_ILA(paOutput, referenceInput, cfg)
%TRAINOFFLINENNDPD_ILA Train a NN-DPD using indirect learning architecture.
%
%   Version « v2 » : ajoute un jeu de validation tenu a l'ecart, un arret
%   anticipe avec patience, la restauration du meilleur etat et une
%   decroissance du taux d'apprentissage. Le budget d'epoques cesse d'etre
%   un reglage a deviner : il est determine par les donnees.
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
    A = [Xn; ones(1,size(Xn,2))].';
    lambda = 1e-4;
    W = (A'*A + lambda*eye(size(A,2))) \ (A'*T.');
    model.fallbackW = W;
    model.net = [];
    model.trainingLoss = mean(sum((A*W - T.').^2,2));
    return;
end

% --- reglages de la procedure, avec valeurs par defaut sures
valFrac  = getfielddef(cfg.nn, 'validationFraction', 0.20);
patience = getfielddef(cfg.nn, 'earlyStopPatience', 8);
lrFactor = getfielddef(cfg.nn, 'lrDecayFactor', 0.5);
lrPat    = getfielddef(cfg.nn, 'lrDecayPatience', 4);

N = size(Xn,2);
perm  = randperm(N);
nVal  = max(1, round(valFrac*N));
idVal = perm(1:nVal);  idTr = perm(nVal+1:end);
Xtr = Xn(:,idTr); Ttr = T(:,idTr);
dlXv = dlarray(single(Xn(:,idVal)), 'CB');
dlTv = dlarray(single(T(:,idVal)), 'CB');

net = createRVTDNN(size(Xn,1), cfg);
avgGrad = []; avgSqGrad = []; iter = 0;
Ntr = size(Xtr,2);
mb  = min(cfg.nn.miniBatchSize, Ntr);
lr  = cfg.nn.learningRateOffline;

bestLoss = inf; bestNet = net; bestEpoch = 0;
sansAmelioration = 0; sansAmeliorationLR = 0;
valHistory = []; lossHistory = [];

for epoch = 1:cfg.nn.maxEpochsOffline
    order = randperm(Ntr);
    for start = 1:mb:Ntr
        iter = iter + 1;
        ids = order(start:min(start+mb-1,Ntr));
        dlX = dlarray(single(Xtr(:,ids)), 'CB');
        dlT = dlarray(single(Ttr(:,ids)), 'CB');
        [loss, gradients] = dlfeval(@localModelLoss, net, dlX, dlT);
        [net, avgGrad, avgSqGrad] = adamupdate(net, gradients, avgGrad, avgSqGrad, iter, lr);
        lossHistory(end+1) = double(gather(extractdata(loss))); %#ok<AGROW>
    end
    % --- perte de validation, sur des paires jamais vues par l'optimiseur
    vLoss = double(gather(extractdata(localModelLoss(net, dlXv, dlTv))));
    valHistory(end+1) = vLoss; %#ok<AGROW>

    if vLoss < bestLoss * (1 - 1e-4)
        bestLoss = vLoss; bestNet = net; bestEpoch = epoch;
        sansAmelioration = 0; sansAmeliorationLR = 0;
    else
        sansAmelioration = sansAmelioration + 1;
        sansAmeliorationLR = sansAmeliorationLR + 1;
        if sansAmeliorationLR >= lrPat
            lr = lr * lrFactor; sansAmeliorationLR = 0;
        end
        if sansAmelioration >= patience, break; end
    end
end

model.net = bestNet;                 % on garde le MEILLEUR etat, pas le dernier
% --- Jeu d'ancrage : les paires de validation, jamais vues par l'optimiseur
% hors ligne, ne serviront JAMAIS non plus aux mises a jour en ligne. Elles
% mesurent la generalisation du reseau sur le PA nominal, ce qu'une mise a
% jour ne doit pas degrader.
nAnc = min(4096, nVal);
model.anchor.Xn = Xn(:, idVal(1:nAnc));
model.anchor.T  = T(:,  idVal(1:nAnc));
% Signal d'ancrage de bout en bout : un segment du signal de reference,
% que la boucle fera traverser predistorteur + PA nominal pour mesurer le
% NMSE reel avant / apres chaque mise a jour.
nSig = min(32768, numel(referenceInput));
model.anchor.xRef = referenceInput(1:nSig);   % 8 blocs : lisse le bruit inter-blocs
model.trainingLoss = lossHistory;
model.validationLoss = valHistory;
model.bestEpoch = bestEpoch;
model.epochsRun = numel(valHistory);
end

function v = getfielddef(s, f, d)
if isfield(s, f), v = s.(f); else, v = d; end
end

function [loss, gradients] = localModelLoss(net, dlX, dlT)
dlY = forward(net, dlX);
loss = mean(sum((dlY - dlT).^2,1));
if nargout > 1
    gradients = dlgradient(loss, net.Learnables);
end
end
