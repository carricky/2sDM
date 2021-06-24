%% load data
load('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/output/extension/all_response.mat')
all_response_hcp = all_response;
load('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/output/ucla/all_response.mat')
all_response_ucla = all_response;

%% remove missing nodes
all_response_hcp = squeeze(all_response_hcp(:, 1, :));
missing_nodes = [249, 239, 243, 129, 266, 109, 115, 118, 250];
all_response_hcp(missing_nodes, :)=[];
all_response_ucla(missing_nodes, :)=[];

%% correlate data
find(all_response_hcp==0)
find(all_response_ucla==0)
imagesc(corr(all_response_hcp, all_response_ucla))