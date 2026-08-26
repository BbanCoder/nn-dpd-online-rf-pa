function ber = computeBER(txBits, rxBits)
%COMPUTEBER Compute BER when bit vectors are available.
if nargin < 2 || isempty(txBits) || isempty(rxBits)
    ber = NaN;
    return;
end
N = min(numel(txBits), numel(rxBits));
ber = mean(txBits(1:N) ~= rxBits(1:N));
end
