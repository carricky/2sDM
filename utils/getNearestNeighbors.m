function [dstsXY, indsXY, sigma] = getNearestNeighbors(X, Y, configParams)
    % [DSTSXY INDSXY SIGMA] = GETNEARESTNEIGHBORS(X, Y, CONFIGPARAMS)
    % calculate affinity matrix between samples X and new samples Y
    
    numberOfPointsX = size(X,1);
    numberOfPointsY = size(Y,1);
    
    [nnData.nnDist,nnData.nnInds] = pdist2(X,Y,'euclidean','Smallest',configParams.kNN);
    nnData.nnDist = nnData.nnDist';
    nnData.nnInds = nnData.nnInds';
    
    
    % Total number of entries in the W matrix
    lTotalSize = configParams.kNN * numberOfPointsY;
    % intialize
    ind = 1;
    rowInds=zeros(1, lTotalSize);
    colInds=zeros(1, lTotalSize);
    vals = zeros(1,lTotalSize);
    if configParams.verbose
        h = waitbar(0,'Calcuating Affinity Matrix for DM coords');
    end
    percentDone = max(round(numberOfPointsY/100),1);
    for i = 1:numberOfPointsY
        % calc the sparse row and column indices
        rowInds(ind : ind + configParams.kNN-1) = nnData.nnInds(i,:);
        colInds(ind : ind + configParams.kNN-1) = i;
        vals(ind : ind + configParams.kNN-1) = nnData.nnDist(i,:);
        ind = ind + configParams.kNN;
        if configParams.verbose  && mod(i,percentDone) == 0
            waitbar(i / numberOfPointsY, h);
        end
    end
    if configParams.verbose
        close(h);
    end
    dstsXY = sparse(rowInds,colInds,vals,numberOfPointsX,numberOfPointsY);
    indsXY = sub2ind([numberOfPointsX,numberOfPointsY], rowInds, colInds);
    % TODO: this function is not implemented fully yet!
    if configParams.self_tune 
        sigma =  (nnData.nnDist(:,configParams.self_tune)+eps)';
    else
        sigma = [];
    end
    return;
