function diffusion_mapX = getDiffusionMapWithExtend(X, Y, configParams)
    % DIFFUSION_MAPX = GETDIFFUSIONMAPWITHEXTEND(X, Y, CONFIGPARAMS)
    % Y is sample set, X is entire dataset
    % first construct diffusion map for Y and then use out-of-sample extension
    % to extend to all points in X
    
    
    if configParams.verbose
        disp('Calculating Diffusion map');
    end
    
    [diffusion_map, Lambda, Psi, nnData] = calcDiffusionMap(Y, configParams);
    inds = 2:size(Psi,2);
    %%
    if size(Y,2) ~= size(X,2) || any(X(:)~=Y(:))
        % If Y is indeed a subset of X, otherwise no need to perform extension
        configParams.sigma0 = 1;
        % Laplacian Pyramid extension
        if configParams.verbose
            disp('Calculating Laplacian Pyramid Extension to all patches');
        end
        PsiBar = zeros(size(X,2), size(Psi,2));
        % calculating for first eigenvector, this can be used as sanity check
        % since all values should be identical as the first eigen-vector is
        % constant
        [PsiBar(:,1), l(1), dstsSqrY, dstsSqrYX, indsYY, indsYX] = calcLaplcianPyramidExtension(configParams, Y, Psi(:,1), X, nnData);
        for i = inds
            [PsiBar(:,i), l(i)] = calcLaplcianPyramidExtension(configParams, Y, Psi(:,i), X, [], dstsSqrY, dstsSqrYX, indsYY, indsYX);
        end
        clear dstsSqrY dstsSqrXY indsXX indsXY
        
        diffusion_mapX = (PsiBar(:,inds).*repmat(Lambda(inds)',size(PsiBar,1),1))';
    else % X==Y
        diffusion_mapX = (Psi(:,inds).*repmat(Lambda(inds)',size(Psi,1),1))';
    end
    
    if configParams.plotResults
        if size(diffusion_mapX,1) > 1
            figure(100);
            scatter3(diffusion_mapX(1,1:10:end),diffusion_mapX(2,1:10:end),diffusion_mapX(3,1:10:end),'d');
            title('Diffusion Map for all data-points')
        else
            figure;
            plot(diffusion_mapX);
            title('Diffusion Map for all data-points')
        end
    end
