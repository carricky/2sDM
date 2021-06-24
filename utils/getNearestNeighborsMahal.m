function [dXY, indsXY, sigma] = getNearestNeighborsMahal(X, Y, configParams)
    % [DSTSXY INDSXY SIGMA] = GETNEARESTNEIGHBORS(X, Y, CONFIGPARAMS)
    % calculate affinity matrix between samples X and new samples Y
    
    numberOfPointsX = size(X,1);
    numberOfPointsY = size(Y,1);
    
    Y = reshape(Y, size(Y, 1), [], configParams.n_sub);
    X = reshape(X, size(X, 1), [], configParams.n_sub);
    
    dXY = sparse(numberOfPointsX, numberOfPointsY);
    for i_sub = 1 : configParams.n_sub
        fprintf('%.1f%% finished \n', i_sub/configParams.n_sub*100)
        X_temp = X(:, :, i_sub);
        Y_temp = Y(:, :, i_sub);
        C = nancov(X_temp) + nancov(Y_temp);
        C = C + diag(2*eps*ones(size(C, 1), 1)); % adding eps to make it positive-definite
        [nnData.nnDist,nnData.nnInds] = pdist2(X_temp, Y_temp, 'mahalanobis',...
            C, 'Smallest',configParams.kNN);
        nnData.nnDist = nnData.nnDist'+eps;
        nnData.nnInds = nnData.nnInds';
        
        
        % Total number of entries in the W matrix
        lTotalSize = configParams.kNN * numberOfPointsY;
        % intialize
        ind = 1;
        rowInds=zeros(1, lTotalSize);
        colInds=zeros(1, lTotalSize);
        vals = zeros(1,lTotalSize);
        
        for i = 1:numberOfPointsY
            % calc the sparse row and column indices
            rowInds(ind : ind + configParams.kNN-1) = nnData.nnInds(i,:);
            colInds(ind : ind + configParams.kNN-1) = i;
            vals(ind : ind + configParams.kNN-1) = nnData.nnDist(i,:);
            ind = ind + configParams.kNN;
        end
        
        dXY = dXY + sparse(rowInds,colInds,vals,numberOfPointsX,numberOfPointsY);
%         indsXY = sub2ind([numberOfPointsX,numberOfPointsY], rowInds, colInds);
    end
    dXY = dXY / configParams.n_sub;
    indsXY = sub2ind([numberOfPointsX,numberOfPointsY], find(dXY));
    if configParams.self_tune
        sigma =  (nnData.nnDist(:,configParams.self_tune)+eps)';
    else
        sigma = [];
    end
    return;
