function [fy, l, dSqrXX, dSqrXY, indsXX, indsXY, sigma0] = ...
        calcLaplcianPyramidExtension(configParams, X, f, Y, dSqrXX, dSqrXY, ...
        indsXX, indsXY)
    % Out-of-sample Laplacian pyramid extension following
    % "Heterogeneous datasets representation and learning using diffusion maps
    % and laplacian pyramids", Rabin and Coifman
    % Here X is sample set, Y is new samples to extend to
    
%     dParams.sigma0 = 25;        % initial coarse sigma
    dParams.maxIters = 15;      % maximum number of iterations
    dParams.errThresh = 0.005;  % error threshold
    dParams.verbose = false;
    dParams.kNN = 50;
    dParams.self_tune = 0;
    if exist('configParams','var') && ~isempty(configParams)
        configParams = setParams(dParams, configParams);
    else
        configParams = dParams;
    end
    
    
    % preparing affinity for sample set X and affinity matrix between
    % sample set X and new samples set Y
    X = X';
    Y = Y';
    if ~exist('dSqrXX', 'var') || isempty(dSqrXX)
        [dX, indsXX] = getNearestNeighbors(X, X, configParams);
%         [dX, indsXX] = getNearestNeighborsMahal(X, X, configParams);
        dSqrXX = dX.^2;
    end

    if ~exist('dSqrXY', 'var') || isempty(dSqrXY)
        %         [dXY, indsXY] = getNearestNeighborsMahal(X, Y, configParams);
        [dXY, indsXY, sigma0] = getNearestNeighbors(X, Y, configParams);
        dSqrXY = dXY.^2;
    end
    if ~exist('sigma0', 'var') || isempty(sigma0)
        sigma0 = median(sqrt(dSqrXX(indsXX)));
    end
    
    %% Laplacian extension following the paper equations
    [S0, s0y] = calcSl(dSqrXX, dSqrXY, sigma0, 0, f, indsXX, indsXY);
    fEst = S0;
    fy = s0y;
    estErr = inf;
    nIters = 0;
    l = 1;
    while estErr > configParams.errThresh && nIters < configParams.maxIters
        dl = f - fEst;
        [Sl, sly] = calcSl(dSqrXX, dSqrXY, sigma0, l, dl, indsXX, indsXY);
        fEst = fEst + Sl;
        fy = fy + sly;
        estErr = norm(fEst - f);
        nIters = nIters + 1;
        l = l + 1;
    end
    
    return;
    
function [Sl, sly] = calcSl(dSqrXX, dSqrXY, sigma0, l, dl, indsXX, indsXY)
    [lIdxsI, lIdxsJ] = ind2sub(size(dSqrXX), indsXX);
    lEntries = exp(-dSqrXX(indsXX) ./ (sigma0.^2/(2^l)));
    Wl =  sparse(lIdxsI, lIdxsJ, lEntries, size(dSqrXX,1), size(dSqrXX,2));
    Wl = (Wl + Wl')/2;
    
    one_over_Ql = spdiags(1 ./ (sum(Wl,2)+eps), 0, size(Wl,1), size(Wl,2));
    Sl = (one_over_Ql*Wl) * dl; % Kl*dl
    [lIdxsI, lIdxsJ] = ind2sub(size(dSqrXY), indsXY);
    lEntries = exp(-dSqrXY(indsXY) ./ (sigma0.^2/(2^l)));
    wly =  sparse(lIdxsI, lIdxsJ, lEntries, size(dSqrXY,1), size(dSqrXY,2));
    d = sum(wly) + eps;
    one_over_ql = spdiags((1 ./ d)', 0, size(wly,2), size(wly,2));
    
    sly = (one_over_ql*wly')*dl;
    return;
