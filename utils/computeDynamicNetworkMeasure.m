function [pc_global, mc_global, pc_region, mc_region] = ...
    computeDynamicNetworkMeasure(data, community, window_size, temporal_label)
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
            temp_conn = coupling(data(:, temporal_label == j_task, i_sub)', window_size);
            temp_time_idx = find(temporal_label == j_task);
            for k_time = 1 : numel(temp_time_idx)
                % PC from positive weights
                P_pos = participation_coef_sign(temp_conn(:, :, k_time), community); 
                pc_region_temp(:, temp_time_idx(k_time)) = P_pos;
                % get the positive temporal connectivity and calculate
                % modularity degree
                temp_conn_pos = temp_conn(:, :, k_time);
                temp_conn_pos = temp_conn_pos .* (temp_conn_pos>0);
                mc_region_temp(:, temp_time_idx(k_time)) = module_degree_zscore(temp_conn_pos, community, 0);
            end
        end
        pc_region(:, :, i_sub) = pc_region_temp;
        mc_region(:, :, i_sub) = mc_region_temp;
    end
%     pc_region = mean(pc_region, 3); % average over subjects
%     mc_region = mean(mc_region, 3); 

    pc_global = mean(pc_region, 1); % average over regions
    mc_global = mean(mc_region, 1);

end
