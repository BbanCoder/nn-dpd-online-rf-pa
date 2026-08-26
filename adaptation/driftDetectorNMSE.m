function flag = driftDetectorNMSE(nmseDb, cfg)
%DRIFTDETECTORNMSE Trigger if NMSE becomes worse than threshold.
if isempty(nmseDb) || ~isfinite(nmseDb)
    flag = false;
    return;
end
flag = nmseDb > cfg.thresholds.NMSE_dB;
end
