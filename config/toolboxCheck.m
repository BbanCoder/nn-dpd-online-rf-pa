function tb = toolboxCheck()
%TOOLBOXCHECK Detect optional MATLAB toolboxes and documented functions.
tb = struct();
tb.fiveG = exist('nrWaveformGenerator','file') == 2 && exist('nrDLCarrierConfig','file') == 2;
tb.deepLearning = exist('dlnetwork','file') == 2 && exist('dlarray','file') == 2;
tb.reinforcementLearning = exist('rlDQNAgent','file') == 2 || exist('rlQAgent','file') == 2;
tb.communications = exist('comm.MemorylessNonlinearity','class') == 8 || exist('qammod','file') == 2;
tb.signalProcessing = exist('pwelch','file') == 2;
end
