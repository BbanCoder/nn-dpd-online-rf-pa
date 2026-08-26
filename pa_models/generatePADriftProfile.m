function profile = generatePADriftProfile(N, cfg, driftType)
%GENERATEPADRIFTPROFILE Generate time-varying PA drift profiles.
if nargin < 3 || isempty(driftType)
    driftType = 'nominal';
end
if N <= 0
    error('N must be positive.');
end
t = linspace(0,1,N).';
M = cfg.pa.memoryDepth;
profile = struct();
profile.G = ones(N,1);
profile.Phi = zeros(N,1);
profile.SatScale = ones(N,1);
profile.MemoryScale = ones(N,M+1);
profile.DriftRate = zeros(N,1);
profile.Type = char(driftType);

driftType = lower(char(driftType));
switch driftType
    case 'nominal'
        % no drift
    case 'thermal'
        % Severe self-heating: gain droop, earlier compression, slow growth
        % of thermal memory effects.
        profile.G = 1 - 0.18*t;
        profile.Phi = deg2rad(10*t);
        profile.SatScale = 1 - 0.22*t;
        for m = 0:M
            profile.MemoryScale(:,m+1) = 1 + 0.15*(m/max(M,1))*t;
        end
        profile.DriftRate = 0.22*ones(N,1);
    case 'powerjump'
        % Abrupt operating-point change: strong compression increase and
        % modified electrical memory after the step.
        stepIdx = max(2, round(0.45*N));
        profile.G(stepIdx:end) = 0.80;
        profile.Phi(stepIdx:end) = deg2rad(18);
        % 0.78 keeps the post-jump PA at the edge of saturation (invertible);
        % lower values drive it beyond saturation where no DPD can linearize.
        profile.SatScale(stepIdx:end) = 0.78;
        for m = 0:M
            profile.MemoryScale(stepIdx:end,m+1) = 1 + 0.20*(m/max(M,1));
        end
        profile.DriftRate(stepIdx:end) = 1;
    case 'memory'
        for m = 0:M
            profile.MemoryScale(:,m+1) = 1 + 0.60*(m/M)*sin(2*pi*(1.5*t + 0.1*m));
        end
        profile.Phi = deg2rad(5*sin(2*pi*t));
        profile.DriftRate = abs([0; diff(profile.MemoryScale(:,end))]);
    case 'aging'
        profile.G = 1 - 0.15*t;
        profile.SatScale = 1 - 0.20*t;
        profile.Phi = deg2rad(8*t.^1.5);
        profile.DriftRate = 0.20*t;
    case 'load'
        profile.G = 1 + 0.10*sin(2*pi*4*t) + 0.06*sin(2*pi*11*t);
        profile.Phi = deg2rad(20*sin(2*pi*3*t));
        profile.SatScale = 1 + 0.10*sin(2*pi*5*t + 0.5);
        profile.DriftRate = abs([0; diff(profile.Phi)]);
    case 'combined'
        profile.G = (1 - 0.22*t) .* (1 + 0.06*sin(2*pi*5*t));
        profile.Phi = deg2rad(12*t + 12*sin(2*pi*2.5*t));
        % End-state saturation stays >= 0.75 so the drifted PA remains
        % invertible: severe but trackable by an adaptive DPD.
        profile.SatScale = 1 - 0.12*t;
        stepIdx = max(2, round(0.55*N));
        profile.SatScale(stepIdx:end) = profile.SatScale(stepIdx:end) * 0.88;
        for m = 0:M
            profile.MemoryScale(:,m+1) = 1 + 0.45*(m/(M+eps))*sin(2*pi*(2*t + 0.12*m));
        end
        profile.DriftRate = abs([0; diff(profile.G)]) + abs([0; diff(profile.Phi)]);
    otherwise
        error('Unknown drift profile: %s', driftType);
end
end
