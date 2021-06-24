% [normed_signal, mean_vector, norm_vector] = normalizeData(pre_signal)
% This function normalizes the input signal to have 0 mean and unit
% variance in time.
% pre_signal: Time x Original Vertices data
% normed_signal: Normalized (Time x Vertices) signal
% mean_vector: 1 x Vertices mean for each time series
% std_vector : 1 x Vertices norm for each time series
function [normed_signal, mean_vector, norm_vector] = normalizeData(pre_signal)
    ones_vector = ones(size(pre_signal, 1), 1) ;
    
    if any(isnan(pre_signal(:))) 
        warning('there are NaNs in the data matrix, synchronization may not work');
    end
    
    pre_signal(isnan(pre_signal)) = 0 ;
    mean_vector = mean(pre_signal, 1) ;
    normed_signal = pre_signal - ones_vector * mean_vector;
    norm_vector = sqrt(sum(normed_signal.^2, 1));
    norm_vector(norm_vector == 0) = 1e-116 ;
    normed_signal = normed_signal ./ (ones_vector * norm_vector) ;
end
