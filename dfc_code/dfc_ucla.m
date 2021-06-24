%% Generates dfc for the healthy control and schz subjects
% add path for dfc estimation
addpath('/Users/siyuangao/Working_Space/fmri/Lindquist_Dynamic_Correlation-master/DCC_toolbox/SWcode/');

% load index for different patient groups
load('/Users/siyuangao/Working_Space/fmri/data/UCLA/idx199.mat')
load('/Users/siyuangao/Working_Space/fmri/data/parcellation_and_network/map268.mat') 
sub_idx_control = idx_control(1:44);
sub_idx_schz = idx_schz(1:10);

missing_nodes = [1, 4, 17, 51, 52, 57, 60, 137, 139, 189, 202];

n_task = 4;
n_sub = numel(sub_idx_schz);
dataset = cell(n_task, 1);
dataset{1} = 'pamenc';
dataset{2} = 'pamret';
dataset{3} = 'scap';
dataset{4} = 'taskswitch';

ucla_task = [];

n_node = 268-numel(missing_nodes);
n_edge = n_node*(n_node-1)/2;
dfc_schz = NaN(n_edge, 1009, n_sub);
start_time = 1;

for i_task = 1 : n_task
    load(['/Users/siyuangao/Working_Space/fmri/data/UCLA/', dataset{i_task}, '199.mat']) 
    all_signal(missing_nodes, :, :) = [];
    n_time = size(all_signal, 2);
    end_time = start_time + n_time - 1;
    time_range = start_time : end_time;
    %     all_signal = region_to_network(all_signal, map268);
    for j_sub = 1 : n_sub
        disp(j_sub)
        temp_dfc = tapered_sliding_window(all_signal(:, :, sub_idx_schz(j_sub))', 22, 3);
        upper_idx = logical(triu(ones(n_node, n_node), 1));
        for k_time = 1 : n_time
            temp_temp_dfc = temp_dfc(:, :, k_time);
            dfc_schz(:, time_range(k_time), j_sub) = temp_temp_dfc(upper_idx);
        end
    end
    start_time = end_time + 1;
end

%% KMeans clustering
non_zero_idx = ~isnan(dfc_schz(1, :, 1));
dfc_schz = permute(dfc_schz(:, non_zero_idx, :), [2, 1, 3]);
dfc_schz = reshape(dfc_schz, size(dfc_schz, 1), []);
% myfunc = @(X,k)(kmeans(X, k, 'replicate', 20, 'Distance', 'cityblock'));
% eva = evalclusters(dfc_schz, myfunc, 'CalinskiHarabasz','KList', 2:10)
c = kmeans(dfc_schz, 5, 'replicate', 1, 'Distance', 'cityblock');

% plot kmeans of dFC on the brain
figure;
scatter3(dm_schz(1, non_zero_idx), dm_schz(2, non_zero_idx), ...
dm_schz(3, non_zero_idx), 40, c, 'filled');
colormap(gca, cmap6)
caxis([0.5, 6.5])