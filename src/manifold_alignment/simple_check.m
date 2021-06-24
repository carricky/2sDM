load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signal_LR_REST_NOGSR.mat')
% load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signal_RL_LANGUAGE_NOGSR.mat')
% load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signal_LR_WM_NOGSR.mat')

% data = cat(1, signal_LR, signal_RL); % (n_voxels, n_frames, n_subs)
data = signal_LR;

n_subs = size(data, 3);

% sub_idx = randperm(n_subs);
% data = data(:, :, 202);

n_subs = size(data, 3);
n_voxel = size(data, 1);
n_frames = size(data, 2);

% zscore the data
for i = 1 : n_subs
    data(:, :, i) = zscore(data(:, :, i), 0, 1);
end

n_d = 10;
dParams.kNN = 270;
dParams.self_tune = 0;
dParams.dist_type = 'euclidean';

% dParams_diffusion.normalization = 'markov';
dParams_diffusion.normalization = 'lb';
dParams_diffusion.t = 1;
dParams_diffusion.verbose = 0;
dParams_diffusion.plotResults = 0;

n_frames = size(data, 2);
c = 1:n_frames;
s = 25*ones(n_frames, 1);
data_ind = data(:, :, 300);
[K, nnData] = calcAffinityMat(data_ind, dParams);
[dm1, Lambda, Psi, Ms, Phi, K_rw] = calcDiffusionMap(K, dParams_diffusion);
figure;scatter3(dm1(1,:),dm1(2,:),dm1(3,:), s, c, 'filled')

% rotate the manifold order
dm1_r = [dm1(:, n_frames/2+1:end), dm1(:, 1:n_frames/2)];
figure;scatter3(dm1_r(1,:),dm1_r(2,:),dm1_r(3,:), s, c, 'filled')

P = dm1/dm1_r;
dm1_a = P*dm1_r;
figure;scatter3(dm1_a(1,:),dm1_a(2,:),dm1_a(3,:), s, c, 'filled')

figure; scatter3([dm1(1,:),dm1_r(1,:)],[dm1(2,:),dm1_r(2,:)], [dm1(3,:),dm1_r(3,:)], 25*ones(1, 2*n_frames), [1*ones(1, n_frames), 2*ones(1,n_frames)], 'filled')
figure; scatter3([dm1(1,:),dm1_a(1,:)],[dm1(2,:),dm1_a(2,:)], [dm1(3,:),dm1_a(3,:)], 25*ones(1, 2*n_frames), [1*ones(1, n_frames), 2*ones(1,n_frames)], 'filled')

% data_ind = data(:, :, 400);
% [K, nnData] = calcAffinityMat(data_ind, dParams);
% [dm2, Lambda, Psi, Ms, Phi, K_rw] = calcDiffusionMap(K, dParams_diffusion);
% figure;scatter3(dm2(1,:),dm2(2,:),dm2(3,:), c, s, 'filled')
% 
% P = dm2/dm1;
% dm3 = P*dm1;
% figure; scatter3([dm3(1,:),dm2(1,:)],[dm3(2,:),dm2(2,:)], [dm3(3,:),dm2(3,:)], 25*ones(1, 2*n_frames), [1*ones(1, n_frames), 2*ones(1,n_frames)], 'filled')
% figure; scatter3([dm1(1,:),dm2(1,:)],[dm1(2,:),dm2(2,:)], [dm1(3,:),dm2(3,:)], 25*ones(1, 2*n_frames), [1*ones(1, n_frames), 2*ones(1,n_frames)], 'filled')