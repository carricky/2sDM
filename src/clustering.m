% % load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/data_nozscore.mat')
% load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/data.mat')
% % load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/good_id_pos_relational.mat')
% missing_nodes = [249, 239, 243, 129, 266, 109, 115, 118, 250];
% % data(missing_nodes, :, :) = [];
% %% brainSync on the data
% num_s = 30;
% % ref_sub = 2;
% for i = 1:num_s
%     if i ~= ref_sub
%         [Y2, ~] = brainSync(data(:, test_range, ref_sub)', data(:, test_range, i)');
%         data(:, test_range, i) = Y2';
%     end
% end
%% clustering the data
k = 4;
rng(665)
[IDX, C, ~, D]= kmeans(dm(:, :)', k);
c = true_label_all(1:size(dm,2));
s = 20*ones(size(dm,2), 1);

figure;subplot(2,2,1);scatter3(dm(1, :), dm(2, :), dm(4, :), s, IDX, 'filled');
colormap(cmap18)
caxis([vmin,vmax])
subplot(2,2,2);scatter3(dm(1, :), dm(2, :), dm(4, :), s, c, 'filled');
colormap(cmap18)
caxis([vmin,vmax])

s = 20*ones(num_t_all, 1);
c = true_label_all(1:num_t_all);
c(train_range) = 3;
subplot(2,2,4);scatter3(dm_all(1, :), dm_all(2, :), dm_all(4, :), s, c, 'filled');
colormap(cmap18)
caxis([vmin,vmax])

%% find representative cluster centers
fix_id = find(dm(1,:)<-0.0277 & dm(2,:)>0.00676 & dm(4,:)>0.003132);
cue_id = find(dm(1,:)>0.016 & dm(2,:)>0.0293 & dm(4,:)>0.00797);
% low_id = find(abs(dm(1,:)-(-2.432722577474538e-04))<1e-6);
% low_id = find(abs(dm(1,:)-(-2.8767e-04))<1e-6);
low_id = find(abs(dm(1,:)-(-0.009512658302650))<1e-6);
% low_id = find(abs(dm(1,:)-(-0.002989968787612))<1e-6);

high_id = find(dm(1,:)>0.009492 & dm(2,:)<-0.02613 & dm(4,:)<-0.001881);
cluster_id = [fix_id, cue_id, low_id, high_id];

% plot the cluster center on the kmeans plot
s = 25*ones(size(dm,2), 1);
c = IDX;
% c(cue_near_idx(1:50)) = 3;
c(cluster_id) = 6;
subplot(2,2,3);scatter3(dm(1, :), dm(2, :), dm(4, :), s, c, 'filled');
colormap(cmap18)
caxis([vmin,vmax])

% get a k-corr plot to determine a good k
k_list = 1:5:600;
[D, I] = pdist2(dm_all(:, test_range)', dm_all(:, train_range)', 'Euclidean', 'Smallest', 600);

% generate the average response
all_response = zeros(268, 2, 4);
figure;
for i_c = 1 : 4
    near_idx = I(:, cluster_id(i_c))+3020;
    corr_list = [];
    for k = k_list
%         ext_response = mean(data(:, near_idx(1:k),ref_sub),2);
        ext_response = mean(mean(data(:, near_idx(1:k),1:num_s),3),2);
        int_response = mean(mean(data(:, cluster_id(i_c),1:num_s),3), 2);
        corr_list = [corr_list;corr(ext_response, int_response)];
    end
    subplot(2,2,i_c);plot(k_list,corr_list)
    
    % generate the neighbor response
    ext_response_temp = mean(mean(data(:, near_idx(1:50),1:num_s),3),2);
    int_response_temp = mean(mean(data(:, cluster_id(i_c),1:num_s),3), 2);
    
    r_idx = ones(268,1);
    r_idx(missing_nodes) = 0;
    r_idx = logical(r_idx);
    
    int_response = zeros(268,1);
    int_response(r_idx) = int_response_temp;
    ext_response = zeros(268,1);
    ext_response(r_idx) = ext_response_temp;
    all_response(:, :, i_c) = [int_response, ext_response];
end


[D_ext, I_ext] = pdist2(dm_all(:, train_range)', dm_all(:, test_range)', 'Euclidean', 'Smallest', 5);
IDX_list = zeros(size(I_ext, 2), 1);
for i = 1 : size(I_ext, 2)
IDX_temp = mode(IDX(I_ext(:, i)));
IDX_list(i) = IDX_temp;
end
for i = 1 : 4
disp(sum(IDX_list==i)/numel(IDX_list))
end


%% !!!figures for the paper
s = 20*ones(num_t_all, 1);
c = zeros(num_t_all, 1);
c(test_range) = 5;
c(train_range) = IDX;
c(cluster_id) = 6;
figure;scatter3(dm_all(1, :), dm_all(2, :), dm_all(4, :), s, c, 'filled');
% figure;scatter3(dm_all(1, :), dm_all(2, :), dm_all(3, :), s, c, 'filled');
colormap(cmap6)
caxis([0.5, 6.5])
hcb=colorbar;
set(hcb,'YTick',[])

%% plots utils
% s = 25*ones(size(dm,2), 1);
% c = IDX;
% % c(cue_near_idx(1:50)) = 3;
% c(1956) = 5;
% figure;scatter3(dm(1, :), dm(2, :), dm(4, :), s, c, 'filled');
% colormap(cmap)
% caxis([vmin,vmax])
%% old
% c_all = [];
% for i = 1 : k
%     temp = mean(mean(data(:, IDX==i,:),3), 2);
%     c_all = [c_all,temp];
% end