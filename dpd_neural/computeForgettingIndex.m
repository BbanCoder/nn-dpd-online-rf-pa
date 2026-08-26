function forgetting = computeForgettingIndex(metricBefore, metricAfter)
%COMPUTEFORGETTINGINDEX Quantify degradation on previous states after adaptation.
if isempty(metricBefore) || isempty(metricAfter)
    forgetting = NaN;
    return;
end
forgetting = metricAfter - metricBefore;
end
