root_path = '/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/output';

%% HCP t_test
load(fullfile(root_path, 'hcp/allsub390.mat'))
n_cohort = length(all_stationary_p_rs);
n_cluster = size(all_stationary_p_rs{1}, 2);

p_all_dist = zeros(n_cluster, 1);
for i_c = 1 : n_cluster
    [h, p_all_dist(i_c)] = ttest2(all_stationary_p_rs{1}(:, i_c), ...
        all_stationary_p_rs{2}(:, i_c));
end

p_all_entropy = zeros(n_cluster, 1);
for i_c = 1 : n_cluster
    [h, p_all_entropy(i_c)] = ttest2(rs_entropy_all{1}(:, i_c), ...
        rs_entropy_all{2}(:, i_c));
end
        
%% UCLA anova
load(fullfile(root_path, 'ucla/allsub.mat'))
n_cohort = length(all_stationary_p_rs);
n_cluster = size(all_stationary_p_rs{1}, 2);

g = cell(size(all_stationary_p_rs{1}, 1), 1);
g(:) = {'adhd'};
g_temp = cell(size(all_stationary_p_rs{2}, 1), 1);
g_temp(:) = {'bpad'};
g = [g;g_temp];
g_temp = cell(size(all_stationary_p_rs{3}, 1), 1);
g_temp(:) = {'hc'};
g = [g;g_temp];
g_temp = cell(size(all_stationary_p_rs{4}, 1), 1);
g_temp(:) = {'schz'};
g = [g;g_temp];

p_all_dist = zeros(n_cluster, 1);
for i_c = 1 : n_cluster  
        
    y = [all_stationary_p_rs{1}(:, i_c);all_stationary_p_rs{2}(:, i_c);...
        all_stationary_p_rs{3}(:, i_c);all_stationary_p_rs{4}(:, i_c)];
    
    [p_all_dist(i_c),tbl] = anova1(y, g);
end
     
p_all_entropy = zeros(n_cluster, 1);
for i_c = 1 : n_cluster  
        
    y = [rs_entropy_all{1}(:, i_c);rs_entropy_all{2}(:, i_c);...
        rs_entropy_all{3}(:, i_c);rs_entropy_all{4}(:, i_c)];
    
    [p_all_entropy(i_c),tbl] = anova1(y, g);
end
     