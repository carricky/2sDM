function [pc_global, pc_region] = ...
    computeDynamicNetworkMeasureDynamicCommunity(data, window_size, temporal_label)
    if ~exist('window', 'var')
        window_size = 15;
    end
    [n_rg, n_time, n_sub] = size(data);
    if ~exist('temporal_label', 'var')
        temporal_label = ones(1, n_time);
    end
    task_list = unique(temporal_label);
    pc_region = zeros(n_rg, n_time, n_sub);
    mc_region = zeros(n_rg, n_time, n_sub);
    parfor i_sub = 1 : n_sub
        fprintf('%d-th subject finished Participation Coefficient\n', i_sub)
        pc_region_temp = zeros(n_rg, n_time);
        mc_region_temp = zeros(n_rg, n_time);
        for j_task = task_list
            %         temp_conn = MTD(data(:, task_idx(i, 1):task_idx(i, 2), sub));
            % TODO: change trim to 1
            temp_conn = coupling(data(:, temporal_label == j_task, i_sub)', window_size, 0, 0);
            temp_time_idx = find(temporal_label == j_task);
            for k_time = 1 : numel(temp_time_idx)
                [M,~]=modularity_louvain_und_sign(temp_conn(:, :, k_time));
                % PC from positive weights
                P_pos = participation_coef_sign(temp_conn(:, :, k_time), M); 
                pc_region_temp(:, temp_time_idx(k_time)) = P_pos;
            end
        end
        pc_region(:, :, i_sub) = pc_region_temp;
    end
%     pc_region = mean(pc_region, 3); % average over subjects
%     mc_region = mean(mc_region, 3); 
%     
    pc_global = mean(pc_region, 1); % average over regions

end
