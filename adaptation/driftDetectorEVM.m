function flag = driftDetectorEVM(evmPercent, cfg)
%DRIFTDETECTOREVM Trigger if EVM exceeds threshold.
if isempty(evmPercent) || ~isfinite(evmPercent)
    flag = false;
    return;
end
flag = evmPercent > cfg.thresholds.EVM_percent;
end
