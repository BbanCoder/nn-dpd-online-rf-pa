function paprDb = computePAPR(x)
%COMPUTEPAPR Compute peak-to-average power ratio in dB.
if nargin < 1 || isempty(x)
    error('Input signal is empty.');
end
x = x(:);
pavg = mean(abs(x).^2);
ppeak = max(abs(x).^2);
paprDb = 10*log10(ppeak / max(pavg, eps));
end
