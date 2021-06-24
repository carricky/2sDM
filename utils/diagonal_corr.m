xPos = [-0.0131015968749571,-0.0252337493620171];
yPos = [0.00262663854949627,0.0173853426157185];

%% diffusion maps
% d_vec = y.Position(1:2) - x.Position(1:2);
% dm_vec = dm(1:2, :)-x.Position(1:2)';
d_vec = yPos - xPos;
dm_vec = dm(1:2, :)-xPos(1:2)';

d_proj = d_vec * dm_vec(1:2, :);
[r, p] = corr(pc_global(:, pc_global~=0)', d_proj(:, pc_global~=0)')

%% plot
x_plot = d_proj(:, pc_global~=0);
y_plot = pc_global(:, pc_global~=0);
figure;scatter(x_plot, y_plot, 20, 'filled', 'r')

coefficients = polyfit(x_plot, y_plot, 1);
xFit = linspace(min(x_plot), max(x_plot), 1000);
yFit = polyval(coefficients , xFit);
hold on;
plot(xFit, yFit, 'b-', 'LineWidth', 5);
% grid on;
% set(gca,'XTick',[]);set(gca,'YTick',[])

%% PCA
xPos = [11.5212097559255,-31.6782676302479];
yPos = [-13.0304083590430,24.6950706208978];

d_vec = yPos(1:2) - xPos(1:2);
score_vec = score(:, 1:2)-xPos(1:2);
d_proj = d_vec * (score_vec(:, 1:2)');
[r, p] = corr(pc_global(:, pc_global~=0)', d_proj(:, pc_global~=0)')

%% plot
x_plot = d_proj(:, pc_global~=0);
y_plot = pc_global(:, pc_global~=0);
figure;scatter(x_plot, y_plot, 20, 'filled', 'r')

coefficients = polyfit(x_plot, y_plot, 1);
xFit = linspace(min(x_plot), max(x_plot), 1000);
yFit = polyval(coefficients , xFit);
hold on;
plot(xFit, yFit, 'b-', 'LineWidth', 5);
% grid on;
% set(gca,'XTick',[]);set(gca,'YTick',[])