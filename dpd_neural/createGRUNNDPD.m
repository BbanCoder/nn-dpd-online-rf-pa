function net = createGRUNNDPD(inputSize, cfg)
%CREATEGRUNNDPD Create a GRU-compatible placeholder model.
% For reproducible R2024 execution in vector-feature mode, this function returns
% the same RVTDNN structure. It preserves the public API for GRU experiments.
net = createRVTDNN(inputSize, cfg);
end
