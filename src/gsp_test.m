% % UCLA
% % ref_sub = 122;
% % % rest
% % % load('/Users/siyuangao/Working_Space/fmri/data/UCLA/rest199.mat')
% % % missing_nodes = [249, 239, 243, 129, 266, 109, 115, 118, 250];
% % % all_signal(missing_nodes, :, :) = [];
% % tar_data = all_signal(:, :, ref_sub);
% % tar_data = zscore(tar_data, 0, 1);
% % n_tp_test = size(tar_data, 2);
% 
% % task
load('/Users/siyuangao/Working_Space/fmri/data/UCLA/all_task.mat')
load('/Users/siyuangao/Working_Space/fmri/data/UCLA/all_label.mat')
missing_nodes = [249, 239, 243, 129, 266, 109, 115, 118, 250];
all_task(missing_nodes, :, :) = [];

train_data = all_task;
% test_task = 4;
% tar_data = all_task(:, label==test_task, 1);
% n_tp_test = sum(label==test_task);

n_rg = size(all_task, 1);
n_tp_train = size(all_task, 2);
n_sub = size(all_task, 3);

n_dim = 10;

V_set = zeros(n_rg, n_dim, n_sub);
for i_sub = 1 : n_sub
    fprintf('computing GSP for %dth sub\n', i_sub)
    % some GSP stuff
    % zscore by region 
    x = train_data(:, :, i_sub)';
    x = zscore(x);
    [A, p] = corr(x);
    A(p>0.05) = 0;
    L = diag(sum(A, 2)) - A;
    [V, D] = eig(L); % eigendecomposition on L
    [~, order] = sort(diag(D), 'ascend'); 
    V = V(:, order);
    V_low = V(:, 1:n_dim);
    
    x_tilda = V_low'*x';
    
    V_set(:, :, i_sub) = V_low;
end

match_time_idx = zeros(n_tp_train, n_sub, n_sub);
match_label_idx = zeros(n_tp_train, n_sub, n_sub);
train_label = true_label_all(773:3020);
accuracy = zeros(n_sub, n_sub);
for i_sub = 1 : n_sub
    fprintf('computing GSP for %dth sub\n', i_sub)
    % some GSP stuff
    % zscore by region 
    x = train_data(:, :, i_sub)';
    x = zscore(x);
    x_tilda_self = V_set(:, :, i_sub)'*x';
    for j_sub = 1 : n_sub
        x_tilda_other = V_set(:, :, j_sub)'*x';
        [D, match_time_idx(:, i_sub, j_sub)] = pdist2(x_tilda_self',x_tilda_other','cosine', 'Smallest', 1);
        match_label_idx(:, i_sub, j_sub) = label(match_time_idx(:, i_sub, j_sub));
    end
    
    for j_sub = 1 : n_sub
        accuracy(i_sub, j_sub) = sum(match_label_idx(:, i_sub, j_sub) == match_label_idx(:, i_sub, i_sub)) / size(match_label_idx, 1);
    end
end

figure;imagesc(accuracy)