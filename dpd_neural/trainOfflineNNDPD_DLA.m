function model = trainOfflineNNDPD_DLA(paInput, referenceInput, cfg)
%TRAINOFFLINENNDPD_DLA Experimental direct-learning placeholder.
% In this simulation package the DLA API is provided and maps to ILA for stable
% execution unless a differentiable PA surrogate is added by the user.
model = trainOfflineNNDPD_ILA(paInput, referenceInput, cfg);
model.type = 'NN_DPD_DLA_placeholder_using_ILA';
end
