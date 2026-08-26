function [trainInput, trainTarget, replayBuffer] = replayBufferUpdate(replayBuffer, newInput, newTarget, cfg)
%REPLAYBUFFERUPDATE Combine new samples with replay samples to limit forgetting.
%   Two replay mechanisms:
%   1. Uniform replay of recent pairs (ratio cfg.adaptation.replayRatio).
%   2. Peak-preserving replay: windows around the largest-amplitude samples
%      seen so far are always re-injected. High-amplitude samples are rare,
%      so uniform mini-batches barely contain them and online updates would
%      otherwise distort the network response in the peak region (the block
%      containing the waveform crest then collapses by several dB).
newInput = newInput(:); newTarget = newTarget(:);
if nargin < 1 || isempty(replayBuffer) || ~isfield(replayBuffer,'input')
    replayBuffer = struct('input',[],'target',[]);
end
if ~isfield(replayBuffer,'peakIn')
    replayBuffer.peakIn = {};
    replayBuffer.peakTgt = {};
    replayBuffer.peakVal = [];
end
nNew = numel(newInput);
nReplay = min(numel(replayBuffer.input), round(cfg.adaptation.replayRatio*nNew));
if nReplay > 0
    ids = randperm(numel(replayBuffer.input), nReplay);
    trainInput = [newInput; replayBuffer.input(ids)];
    trainTarget = [newTarget; replayBuffer.target(ids)];
else
    trainInput = newInput;
    trainTarget = newTarget;
end

% Update the peak-window store with this block's crest.
win = 128;
maxSegs = 8;
[pkVal, pkIdx] = max(abs(newTarget));
lo = max(1, pkIdx - win/2);
hi = min(nNew, pkIdx + win/2 - 1);
replayBuffer.peakIn{end+1} = newInput(lo:hi);
replayBuffer.peakTgt{end+1} = newTarget(lo:hi);
replayBuffer.peakVal(end+1) = pkVal;
if numel(replayBuffer.peakVal) > maxSegs
    [~, order] = sort(replayBuffer.peakVal, 'descend');
    keep = sort(order(1:maxSegs));
    replayBuffer.peakIn = replayBuffer.peakIn(keep);
    replayBuffer.peakTgt = replayBuffer.peakTgt(keep);
    replayBuffer.peakVal = replayBuffer.peakVal(keep);
end
% Always append the stored peak windows to the training set.
trainInput = [trainInput; vertcat(replayBuffer.peakIn{:})];
trainTarget = [trainTarget; vertcat(replayBuffer.peakTgt{:})];

maxBuffer = max(4*nNew, cfg.adaptation.minSamples);
replayBuffer.input = [replayBuffer.input(:); newInput];
replayBuffer.target = [replayBuffer.target(:); newTarget];
if numel(replayBuffer.input) > maxBuffer
    replayBuffer.input = replayBuffer.input(end-maxBuffer+1:end);
    replayBuffer.target = replayBuffer.target(end-maxBuffer+1:end);
end
end
