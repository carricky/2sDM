addpath('../utils/');

% load the data
dir = 'xxx'; % path to the data
load(dir);


% config the parameters
configAffParams1.dist_type = 'euclidean';
configAffParams1.kNN = 270;
configAffParams1.self_tune = 0;

configDiffParams1.t = 1;

configAffParams2 = configAffParams1;
configDiffParams2 = configDiffParams1;

n_d = 5;


% run the code
[dm] = calc2sDM(data, n_d, configAffParams1, configAffParams2, configDiffParams1, configDiffParams2);


% visualize the first three non-trivial dimensions
figure;scatter3(dm(1, :), dm(2, :), dm(3, :));