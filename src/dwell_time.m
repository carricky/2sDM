addpath('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/utils/BrainSync/')
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/data.mat')
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/true_label/true_label_all_with_rest.mat')
load('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/utils/cmap18.mat')

% choose less subjects for quicker computation
data = data(:, :, 1:30);

% task_endtime = [772,1086,1554,2084,2594,3020]
% task_length = [772,314,468,530,510,426]
test_range = 3021:3270;
train_range = 1:3020;

test_data = data(:, test_range, :);
num_t_test = size(test_data, 2);

train_data = data(:, train_range, :);
num_s = size(train_data, 3);
num_t_train = size(train_data, 2);
num_r = size(train_data, 1);
num_t_all = num_t_test+num_t_train;

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
[dm, ~, embed, lambda1, lambda2, sigma1, sigma2] = calc2sDM(train_data, n_d, configAffParams1, configAffParams2, configDiffParams1, configDiffParams2);

s = 20*ones(num_t_train, 1);
c = true_label_all(train_range);
vmin = -0.5;
vmax = 17.5;

% figure;subplot(2,2,1);scatter3(dm(1, :), dm(2, :), dm(3, :), s, c, 'filled');
% colormap(cmap)
% caxis([vmin,vmax])
% 
% subplot(2,2,2);scatter3(dm(1, :), dm(2, :), dm(4, :), s, c, 'filled');
% colormap(cmap)
% caxis([vmin,vmax])

%% cluster the tasks
k = 4;
IDX = kmeans(dm(:, :)', k);

%% generate extension embedding
sub_list = 1:num_s;
dwell_time_list = zeros(numel(sub_list), 4);
IDX_list_all = zeros(numel(sub_list), numel(test_range));
for ref_sub = sub_list
    disp(ref_sub)
%     figure;
% ref_sub = 1;
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
    
    [D_ext, I_ext] = pdist2(dm_all(:, train_range)', dm_all(:, test_range)', 'Euclidean', 'Smallest', 10);
    IDX_list = zeros(size(I_ext, 2), 1);
    for i = 1 : size(I_ext, 2)
        IDX_temp = mode(IDX(I_ext(:, i)));
        IDX_list(i) = IDX_temp;
    end
    for i = 1 : 4
        dwell_time_list(ref_sub, i) = (sum(IDX_list==i)/numel(IDX_list));
    end
    IDX_list_all(ref_sub, :) = IDX_list;
    disp(dwell_time_list(ref_sub, :));
    
end



%% clustering the data
c = true_label_all(1:size(dm,2));
s = 20*ones(size(dm,2), 1);

figure;subplot(2,2,1);scatter3(dm(1, :), dm(2, :), dm(4, :), s, IDX, 'filled');
subplot(2,2,2);scatter3(dm(1, :), dm(2, :), dm(4, :), s, c, 'filled');
colormap(cmap)
caxis([vmin,vmax])


