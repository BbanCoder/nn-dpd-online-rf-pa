function doUpdate = periodicUpdatePolicy(blockIndex, periodBlocks)
%PERIODICUPDATEPOLICY Periodic recalibration trigger.
if nargin < 2 || isempty(periodBlocks)
    periodBlocks = 5;
end
doUpdate = mod(blockIndex, periodBlocks) == 0;
end
