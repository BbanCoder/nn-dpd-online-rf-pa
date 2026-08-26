function evmPercent = computeEVM(y, ref)
%COMPUTEEVM EVM in percent after complex gain alignment. Compatible with R2024.
y = y(:); ref = ref(:);
if isempty(y) || isempty(ref), error('Signals must be non-empty.'); end
N = min(numel(y), numel(ref));
y = y(1:N); ref = ref(1:N);
g = (y' * ref) / max(y' * y, eps);
yA = g * y;
errRms = sqrt(mean(abs(ref - yA).^2));
refRms = sqrt(mean(abs(ref).^2));
evmPercent = 100 * errRms / max(refRms, eps);
end
