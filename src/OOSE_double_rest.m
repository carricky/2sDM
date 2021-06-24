%% add path
addpath('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/src')
addpath('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/utils/BrainSync/')
addpath('/Users/siyuangao/Working_Space/fmri/bctnet/BCT/2017_01_15_BCT/')

%% load data
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/data.mat')
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/true_label/true_label_all_withrest_coarse.mat')
load('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/utils/cmap10.mat')
load('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/utils/cmap4_2.mat')
load('/Users/siyuangao/Working_Space/fmri/data/parcellation_and_network/map259.mat')

% choose less subjects for quicker computation
data = data(:, :, 1:30);

% % zscore the region
% for i_sub = 1 : size(data,3)
%     data(:, :, i_sub) = zscore(data(:, :, i_sub), 0, 2);
% end

task_endtime = [0,772,1086,1554,2084,2594,3020,4176];
% task_length = [772,314,468,530,510,426]
test_range = 3021:5382;
train_range = 1:3020;

test_data = data(:, test_range, :);
num_t_test = size(test_data, 2); % length of testing data

train_data = data(:, train_range, :); % task data to generate manifold
num_s = size(train_data, 3); % number of subjects
num_t_train = size(train_data, 2); % length of training data
num_r = size(train_data, 1); % number of regions
num_t_all = num_t_test+num_t_train; % total length of time


%% config the parameters
k = 500;
configAffParams1.dist_type = 'euclidean';
configAffParams1.kNN = k;
configAffParams1.self_tune = 0;

configDiffParams1.t = 1;
configDiffParams1.normalization='lb';

configAffParams2 = configAffParams1;
configDiffParams2 = configDiffParams1;

n_d = 7;

%% generate training embedding
% embed is the n_t*(n_sub*n_d) first round flatten matrix, dm is the final embedding
[dm, ~, embed, lambda1, lambda2, sigma1, sigma2] = calc2sDM(train_data,...
    n_d, configAffParams1, configAffParams2, configDiffParams1, configDiffParams2); 

c = true_label_all(train_range);
vmin = -0.5;
vmax = 9.5;

figure;subplot(2,2,1);scatter3(dm(1, :), -dm(2, :), dm(3, :), 20, c, 'filled');
colormap(cmap10)
caxis([vmin,vmax])

subplot(2,2,2);scatter3(dm(1, :), -dm(2, :), dm(4, :), 20, c, 'filled');
colormap(cmap10)
caxis([vmin, vmax])

%% kmeans part
rng(665)
n_cluster = 4;
kmeans_idx = kmeans(dm(1:4, :)', n_cluster, 'Replicates', 100);
kmeans_idx(kmeans_idx==1)=8;
kmeans_idx(kmeans_idx==2)=6;
kmeans_idx(kmeans_idx==3)=5;
kmeans_idx(kmeans_idx==4)=7;

% kmeans_idx(kmeans_idx==1)=6;
% kmeans_idx(kmeans_idx==2)=8;
% kmeans_idx(kmeans_idx==3)=7;
% kmeans_idx(kmeans_idx==4)=5;
kmeans_idx = kmeans_idx-4;



figure;
scatter3(dm(1, :), -dm(2, :), dm(4, :), 20, kmeans_idx, 'filled');
colormap(cmap4)
caxis([0.5, 5.5])

%% generate extension embedding
% for ref_sub = 1 : 9
%     figure;
ref_sub = 1;
psi1 = zeros(num_t_test, n_d*num_s);
configAffParams3 = configAffParams1;
configAffParams3.kNN = 500;
% ref_sub = 2;
for i = 1 : num_s
    disp(i)
    %     [K] = calcAffinityMat2(wm_data(:,:,i), data(:,:,i), k, sigma1(i));
    data_ind = zeros(num_r, num_t_all);
    data_ind(:, train_range) = train_data(:, :, i);
    data_ind(:, test_range) = test_data(:, :, i);
    if i ~= ref_sub
        [Y2, R] = brainSync(test_data(:, :, ref_sub)', data_ind(:, test_range)');
        data_ind(:, test_range) = Y2';
    end
    configAffParams3.sig = sigma1(i);
    [K] = calcAffinityMat(data_ind, configAffParams3);
    K = K(test_range, train_range);
    K = K./sum(K, 2);
    psi1(:, (i-1)*n_d+1:i*n_d) = K*embed(:, (i-1)*n_d+1:i*n_d)./lambda1(:, i)';
end

% second round embedding

embed_all = zeros(n_d*num_s, num_t_all);
embed_all(:, train_range) = embed';
embed_all(:, test_range) = psi1';
configAffParams3.sig = sigma2;
[K, ~] = calcAffinityMat(embed_all, configAffParams3);
K = K(test_range, train_range);
K = K./sum(K, 2);
psi2 = K*dm'./lambda2';

dm_all = zeros(n_d, num_t_all);
dm_all(:, train_range) = dm;
dm_all(:, test_range) = psi2';

% plot
c = zeros(num_t_all, 1);
c(train_range) = kmeans_idx;
c(test_range) = 5;
vmin = 0.5;
vmax = 5.5;

subplot(2,2,3);scatter3(dm_all(1, :), -dm_all(2, :), dm_all(3, :), 20, c, 'filled');
colormap(cmap4)
caxis([vmin,vmax])

subplot(2,2,4);
f=scatter3(dm_all(1, train_range), -dm_all(2, train_range), dm_all(4, train_range), 20, c(train_range), 'filled');
f.MarkerFaceAlpha = 0.45;
hold on
f=scatter3(dm_all(1, test_range), -dm_all(2, test_range), dm_all(4, test_range), 20, c(test_range), 'filled');
f.MarkerFaceAlpha = 1;
colormap(cmap4)
caxis([vmin,vmax])
ax = gca;
ax.FontSize = 16;
% end

%% Brain state transition
% get rs_kmeans_idx
[~, I] = pdist2(dm(1:4, :)', dm_all(1:4, test_range)', 'euclidean','Smallest', 1);
rs_kmeans_idx = kmeans_idx(I);
kmeans_idx_all = [kmeans_idx; rs_kmeans_idx];

n_task = 6;

trans_mat = zeros(n_task, n_cluster, n_cluster);
stationary_p = zeros(n_task, n_cluster);
stateNames = ["High cog" "Transition" "Fixation" "Low cog"];
for i_task = 1 : n_task
    for j_tp = task_endtime(i_task)+1 : task_endtime(i_task+1)-1
        trans_mat(i_task, kmeans_idx_all(j_tp), kmeans_idx_all(j_tp+1))...
            = trans_mat(i_task, kmeans_idx_all(j_tp), kmeans_idx_all(j_tp+1)) + 1;
    end
    temp = squeeze(trans_mat(i_task, :, :));
    trans_mat(i_task, :, :) =  temp ./ sum(temp, 2);
    [temp, ~] = eigs(squeeze(trans_mat(i_task, :, :))');
    stationary_p(i_task, :) = temp(:, 1) / sum(temp(:, 1));

%     subplot(2,3,i_task)
    figure;
    
    %     heatmap(squeeze(trans_mat(i_task, :, :)), 'Colormap', parula)
    %     caxis([0, 1])
    
    temp = squeeze(trans_mat(i_task,:,:));
    temp(temp<0.03) = 0; % removing small edges
    if temp(2,3)==0 && temp(3,2)==0
        temp(2,3) = 0.0001;
    end
    mc = dtmc(temp,'StateNames',stateNames);
    %     h = graphplot(mc,'ColorEdges',true, 'LabelEdges', true);
    h = graphplot(mc,'ColorEdges',true);
    
    %     temp = temp';
    %     h.LineWidth=log(temp(temp~=0)+1.3)*10;
    h.LineWidth = 2;
    %     h.NodeFontSize = 0;
    %     h.EdgeFontSize = 10;
    %     h.NodeLabelColor = [1, 1, 1];
    h.NodeLabel = {};
    
%     Cdata = colormap(flipud(gray)); % node color
    temp_colormap = [[0, 0, 0];flipud(autumn)]; % edge color
    colormap(temp_colormap)
    %     colormap(flipud(autumn))
    %     colormap(jet)
    
    %     Cdata = colormap(flipud(autumn));
%     colormap(flipud(gray))
%     caxis([-0.5, 1])
    
%     temp_p = stationary_p(i_task, :);
%     c_idx = floor(temp_p ./ 0.6 .* 256);
%     h.NodeColor = Cdata(c_idx, :);
    colorbar('off')
    set(gca,'color','k');
end

% WM blocks transition
% 0back
temporal_idx = find(true_label_all==2);
trans_mat_0back = zeros(n_cluster, n_cluster);
for i_tp = 1 : numel(temporal_idx)-1
%     if temporal_idx(i_tp) == temporal_idx(i_tp+1)-1
        c1 = kmeans_idx_all(temporal_idx(i_tp));
        c2 = kmeans_idx_all(temporal_idx(i_tp+1));
        trans_mat_0back(c1, c2) = trans_mat_0back(c1, c2) + 1;
%     end
end
trans_mat_0back = trans_mat_0back./ sum(trans_mat_0back, 2);
figure;
temp = trans_mat_0back;
temp(temp<0.01) = 0; % removing small edges
if temp(2,3)==0 && temp(3,2)==0
    temp(2,3) = 0.0001;
end
mc = dtmc(temp,'StateNames',stateNames);
h = graphplot(mc,'ColorEdges',true, 'LabelEdges', true);

temp = temp';
h.LineWidth=log(temp(temp~=0)+1.3)*10;
h.NodeFontSize = 13;
h.EdgeFontSize = 10;

Cdata = colormap(flipud(gray)); % node color
temp_colormap = [[1, 1, 1];flipud(autumn)]; % edge color
colormap(temp_colormap)

[temp, ~] = eigs(trans_mat_0back');
stationary_p_0back = temp(:, 1) / sum(temp(:, 1));
c_idx = floor(stationary_p_0back ./ 0.6 .* 256);
h.NodeColor = Cdata(c_idx, :);
colorbar('off')

% 2back
temporal_idx = find(true_label_all==3);
trans_mat_0back = zeros(n_cluster, n_cluster);
for i_tp = 1 : numel(temporal_idx)-1
%     if temporal_idx(i_tp) == temporal_idx(i_tp+1)-1
        c1 = kmeans_idx_all(temporal_idx(i_tp));
        c2 = kmeans_idx_all(temporal_idx(i_tp+1));
        trans_mat_0back(c1, c2) = trans_mat_0back(c1, c2) + 1;
%     end
end
trans_mat_0back = trans_mat_0back./ sum(trans_mat_0back, 2);
figure;
temp = trans_mat_0back;
temp(temp<0.01) = 0; % removing small edges
if temp(2,3)==0 && temp(3,2)==0
    temp(2,3) = 0.0001;
end
mc = dtmc(temp,'StateNames',stateNames);
h = graphplot(mc,'ColorEdges',true, 'LabelEdges', true);

temp = temp';
h.LineWidth=log(temp(temp~=0)+1.3)*10;
h.NodeFontSize = 13;
h.EdgeFontSize = 10;

Cdata = colormap(flipud(gray)); % node color
temp_colormap = [[1, 1, 1];flipud(autumn)]; % edge color
colormap(temp_colormap)

[temp, ~] = eigs(trans_mat_0back');
stationary_p_2back = temp(:, 1) / sum(temp(:, 1));
c_idx = floor(stationary_p_2back ./ 0.6 .* 256);
h.NodeColor = Cdata(c_idx, :);
colorbar('off')

% plot stationary distribution
figure;
subplot(3,3,1)
b = bar(stationary_p_0back);
b.FaceColor = 'flat';
b.CData(1:4, :) = cmap4(1:4, :);
ylim([0, 0.6])
subplot(3,3,2)
b = bar(stationary_p_2back);
b.FaceColor = 'flat';
b.CData(1:4, :) = cmap4(1:4, :);
ylim([0, 0.6])
for i_task = 1 : n_task
    subplot(3, 3, i_task+2)
    b = bar(stationary_p(i_task, :));
    b.FaceColor = 'flat';
    b.CData(1:4, :) = cmap4(1:4, :);
    ylim([0, 0.6])
end

% calculate skewness
% skew = zeros(n_task+2, 1);
% temp_mean = (1:4) * stationary_p_0back;
% temp_m = ((1:4) - temp_mean).^3 * stationary_p_0back;
% temp_std = sqrt(sum(((1:4) - temp_mean).^2 .* stationary_p_0back'));
% skew(1) = temp_m / temp_std^3;
% temp_mean = (1:4) * stationary_p_2back;
% temp_m = ((1:4) - temp_mean).^3 * stationary_p_2back;
% temp_std = sqrt(sum(((1:4) - temp_mean).^2 .* stationary_p_2back'));
% skew(2) = temp_m / temp_std^3;
% for i_task = 1 : n_task
%     temp_mean = (1:4) * stationary_p(i_task, :)';
%     temp_m = ((1:4) - temp_mean).^3 * stationary_p(i_task, :)';
%     temp_std = sqrt(sum(((1:4) - temp_mean).^2 .* stationary_p(i_task, :)));
%     skew(i_task+2) = temp_m / temp_std^3;
% end

%% dState1 vs dState2
stationary_all = [stationary_p_0back';stationary_p_2back';stationary_p];

x = stationary_all(:, [1, 4]);
figure;
scatter(x(:, 1), x(:, 2), 'filled', 'r');

% chi-square goodness of fit test
cs_all = zeros(8, 1);

% 0back
state_num_tp = zeros(4, 1);
temp_range = find(true_label_all==2);
for j_state = 1 : 4
    state_num_tp(j_state) = sum(kmeans_idx_all(temp_range) == j_state);
end
expected = [numel(temp_range)*0.25, numel(temp_range)*0.25, ...
    numel(temp_range)*0.25, numel(temp_range)*0.25];
[h,p,st] = chi2gof(1:4,'Ctrs',1:4,'Frequency',state_num_tp, ...
    'Expected',expected);
cs_all(1) = st.chi2stat;

% 2back
state_num_tp = zeros(4, 1);
temp_range = find(true_label_all==3);
for j_state = 1 : 4
    state_num_tp(j_state) = sum(kmeans_idx_all(temp_range) == j_state);
end
expected = [numel(temp_range)*0.25, numel(temp_range)*0.25, ...
    numel(temp_range)*0.25, numel(temp_range)*0.25];
[h,p,st] = chi2gof(1:4,'Ctrs',1:4,'Frequency',state_num_tp, ...
    'Expected',expected);
cs_all(2) = st.chi2stat;

% other tasks
for i_task = 1 : n_task
    state_num_tp = zeros(4, 1);
    temp_range = task_endtime(i_task)+1 : task_endtime(i_task+1)-1;
    for j_state = 1 : 4
        state_num_tp(j_state) = sum(kmeans_idx_all(temp_range) == j_state);
    end
    expected = [numel(temp_range)*0.25, numel(temp_range)*0.25, ...
        numel(temp_range)*0.25, numel(temp_range)*0.25];
    [h,p,st] = chi2gof(1:4,'Ctrs',1:4,'Frequency',state_num_tp, ...
                        'Expected',expected);
    cs_all(i_task+2) = st.chi2stat;
end

%% RS transition
temporal_idx = find(true_label_all(1:num_t_all)==9);
trans_mat_rs = zeros(n_cluster, n_cluster);
for i_tp = 1 : numel(temporal_idx)-1
%     if temporal_idx(i_tp) == temporal_idx(i_tp+1)-1
        c1 = kmeans_idx_all(temporal_idx(i_tp));
        c2 = kmeans_idx_all(temporal_idx(i_tp+1));
        trans_mat_rs(c1, c2) = trans_mat_rs(c1, c2) + 1;
%     end
end
trans_mat_rs = trans_mat_rs./ sum(trans_mat_rs, 2);
[temp, ~] = eigs(trans_mat_rs');
stationary_p_rs= temp(:, 1) / sum(temp(:, 1));
figure;
b = bar(stationary_p_rs);
b.FaceColor = 'flat';
b.CData(1:4, :) = cmap4(1:4, :);
ylim([0, 0.6])

temp_mean = (1:4) * stationary_p_rs;
temp_m = ((1:4) - temp_mean).^3 * stationary_p_rs;
temp_std = sqrt(sum(((1:4) - temp_mean).^2 .* stationary_p_rs'));
skew_rs = temp_m / temp_std^3;

%% entropy transition calculation
task_entropy = zeros(7, 4);
figure;
hold on
% colors = [[0,0.23,1];[1,0.75,0];[0.25, 1, 0.75];[0.6, 0, 0.6];[0.65,0.16,0.16];[0.86,0.44,0.58]];
for i_task = 1 : 6
    if i_task <=6
        temp_trans_mat = squeeze(trans_mat(i_task, :, :));
    else
        temp_trans_mat = trans_mat_rs;
    end
    for j_state = 1 : 4
        temp_pmf = temp_trans_mat(j_state,:);
        temp_pmf(temp_pmf==0) = [];
        task_entropy(i_task, j_state) = -sum(temp_pmf.*log(temp_pmf));
    end
    if i_task <= 6
        scatter(1:4, task_entropy(i_task, :), 150, cmap10(i_task+3,:), 'filled')
    else
        scatter(1:4, task_entropy(i_task, :), 'r')
    end
    ylim([0, 1.4])
end
bh = boxplot(rs_entropy, 'Colors', cmap4(5, :))
set(bh, 'LineWidth', 2)
ylim([0, 1.4])
hold off

%% Distance of RS to the other task
js_dist = zeros(9, 9);
stationary_all = [stationary_p_0back';stationary_p_2back';stationary_p; ...
    stationary_p_rs'];
for i_task = 1 : 9
    temp_idx = logical(ones(9, 1));
    temp_idx(i_task) = false;
    js_dist(i_task, temp_idx) = JSDiv(stationary_all(temp_idx, :), ...
        stationary_all(i_task, :));
end
adj = 1 - js_dist;
% adj(adj<mean(adj)) = 0;
adj(1:8, :) = 0;
adj = adj*100;
G = graph(adj, {'0back', '2back', 'WM', 'EMO', 'GAM', 'MOT', 'SOC', ...
    'REL', 'RS'}, 'lower', 'omitselfloops');

% figure; plot(G, 'Layout','force', 'WeightEffect', 'inverse')
figure; plot(G, 'Layout','force', 'EdgeLabel', G.Edges.Weight)

%% color resting-state by particiation coefficient
% align subjects
ref_sub = 1;
rs_aligned = zeros(num_r, num_t_test, num_s);
for i = 1 : num_s
    disp(i)
    rs_aligned(:, :, i) = test_data(:, :, i);
    if i ~= ref_sub
        [Y2, ~] = brainSync(test_data(:, :, ref_sub)', rs_aligned(:, :, i)');
        rs_aligned(:, :, i) = Y2';
    end
end

rs_pc = computeDynamicNetworkMeasure(rs_aligned, map259, 15);
rs_pc(rs_pc==0) = mean(rs_pc(rs_pc~=0));
figure;
scatter3(dm_all(1, test_range), dm_all(2, test_range), dm_all(4, test_range), 20, rs_pc, 'filled');
corr(dm_all(:, test_range)', rs_pc')
colormap(jet)

figure;
scatter(dm_all(1, test_range), rs_pc)

