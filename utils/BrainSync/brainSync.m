%%  BrainSync: Syncronize the subject fMRI data to the reference fMRI data
%   Authors: Anand A Joshi, Minqi Chong, Jian Li, Richard M. Leahy
%   Input:
%       X - Time series of the reference data (Time x Vertex)
%       Y - Time series of the subject data (Time x Vertex)
%       
%   Output:
%       Y2 - Syncronized subject data with respect to reference data (Time x Vertex)
%       R - The orthogonal rotation matrix (Time x Time)
%       
%   Please cite the following publication:
%       AA Joshi, M Chong, RM Leahy, BrainSync: An Orthogonal Transformation for Synchronization of fMRI Data Across Subjects, Proc. MICCAI 2017, in press.
%

function [Y2, R] = brainSync(X, Y)
    [X,mu1,sigma1] = zscore(X,0,1);
    [Y,mu2,sigma2] = zscore(Y,0,1);
    
    if size(X,1) > size(X,2)
%         warning('The input is possibly transposed. Please check to make sure that the input is time x vertices!');
    end
    
    C = X * Y';
    [U, ~, V] = svd(C);
    R = U * V';
    Y2 = R * Y;
    
    Y2 = Y2.*sigma2+mu2;
end
