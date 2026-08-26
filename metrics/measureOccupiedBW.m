function bw = measureOccupiedBW(x, fs)
%MEASUREOCCUPIEDBW 99%-power occupied bandwidth of a complex baseband signal.
%   Used to define the ACPR useful channel from the signal itself, so that
%   the channel masks stay correct even when configuration labels do not
%   match the actual waveform (e.g. the real 5G NR generation path).
x = x(:);
if isempty(x), error('Input signal is empty.'); end
N = numel(x);
P = abs(fftshift(fft(x))).^2;
f = (-floor(N/2):ceil(N/2)-1).' * fs/N;
c = cumsum(P) / max(sum(P), eps);
f1 = f(find(c >= 0.005, 1, 'first'));
f2 = f(find(c >= 0.995, 1, 'first'));
bw = max(f2 - f1, fs/N);
end
