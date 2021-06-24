function [K] = calcAffinityMat2(X, Y, k, sig)
% calculate affinity matrix of data matrix X, size N by M 
% N - length of feature vector, M - number of vectors
% affinity matrix is calculated for kNN nearest neighbors, resulting in
% sparse matrix. This saves on runtime.
% the scale for the affinity matrix can be set using auto-tuning
%
% Gal Mishne

[~,M] = size(X);
[~,N] = size(Y);

%% affinity matrix
% tic
[Dis, Inds] = pdist2(Y',X', 'Euclidean','Smallest',k);
% toc
Dis = Dis';
Inds = Inds';
% calc the sparse row and column indices
rowInds = repmat((1:M),k,1);
rowInds = rowInds(:);
colInds = double(Inds(:));
vals    = Dis(:);

vals = exp(-vals.^2/sig^2);
K = sparse(rowInds, colInds, vals, M, N);

% K = (K + K')/2;
return
