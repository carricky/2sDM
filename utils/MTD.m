function [mtd, dyn_z] = MTD(data, window)
    %time-resolved connectivity - Multiplication of Temporal Derivatives (MTD)
    
    [nNodes,nTime] = size(data);
    %calculate temporal derivative
    td = diff(data');
    
    %functional coupling score
    fc = bsxfun(@times,permute(td,[1,3,2]),permute(td,[1,2,3]));
    
    %simple moving average (need to define window length)
    if ~exist('window', 'var')
        window = 15;
    end
    
    mtd_filter = 1/window*ones(window,1);
    mtd = zeros(nTime,nNodes,nNodes);
    
    for j = 1:nNodes
        for k = 1:nNodes
            mtd(2:end,j,k) = filter(mtd_filter,1,fc(:,j,k));
        end
    end
    
    mtd(1:round(window/2),:,:) = [];
    mtd(round(nTime-window):nTime,:,:) = 0;
    mtd = permute(mtd,[2,3,1]);
    mtd(:,:,1) = mtd(:,:,2);
    
    %time-averaged connectivity matrix
    dyn_avg = nanmean(mtd,3);
    dyn_z = weight_conversion(dyn_avg,'normalize'); %normalize