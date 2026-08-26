function [X, T, validIdx] = makeNNFeatures(inputSignal, targetSignal, cfg)
%MAKENNFEATURES Build real-valued RVTDNN features from complex IQ samples.
inputSignal = inputSignal(:);
if nargin < 2 || isempty(targetSignal)
    targetSignal = [];
else
    targetSignal = targetSignal(:);
    if numel(targetSignal) ~= numel(inputSignal)
        error('targetSignal must have the same length as inputSignal.');
    end
end
if isempty(inputSignal), error('Input signal is empty.'); end
if any(~isfinite(real(inputSignal))) || any(~isfinite(imag(inputSignal)))
    error('Input contains NaN or Inf.');
end
M = cfg.nn.memoryDepth;
N = numel(inputSignal);
validIdx = (M+1:N).';
numFeatures = 4*(M+1); % Re, Im, |x|, |x|^2 for each delay
X = zeros(numFeatures, numel(validIdx));
row = 1;
for m = 0:M
    xm = inputSignal(validIdx-m);
    X(row,:) = real(xm).'; row = row + 1;
    X(row,:) = imag(xm).'; row = row + 1;
    X(row,:) = abs(xm).'; row = row + 1;
    X(row,:) = (abs(xm).^2).'; row = row + 1;
end
if ~isempty(targetSignal)
    tgt = targetSignal(validIdx);
    T = [real(tgt).'; imag(tgt).'];
else
    T = [];
end
end
