% complexity_analysis_independent.m
% 该脚本分析 BA 算法、KKT 条件法和引入惩罚项方法在不同信道维数下的计算复杂度。
% 独立变化输入符号数 N 和输出符号数 M，并绘制三维图表。

clear; clc; close all;
P=[0.5,0.5;0.5,0.5];


%先调用一次函数,防止污染第一个数据
BA_test = arimoto_blahut(P); 
KKT_tst = compute_capacity_KKT(P);
Penal_test = penalty_capacity(P, 100, 1000);

%% 一、定义参数范围
input_symbols = 2:5:12;  % 输入符号数 N 从2到20，每次增加2
output_symbols = 2:5:12; % 输出符号数 M 从2到20，每次增加2
num_N = length(input_symbols);
num_M = length(output_symbols);

% 初始化运行时间存储矩阵
time_BA = zeros(num_N, num_M);
time_KKT = zeros(num_N, num_M);
time_Penalty = zeros(num_N, num_M);

% 定义惩罚参数（用于引入惩罚项方法）
lambda = 100;
mu = 1000;

%% 二、进行复杂度测试
fprintf('开始复杂度测试...\n');
tStart = tic;

for i = 1:num_N
    N = input_symbols(i);
    for j = 1:num_M
        M = output_symbols(j);
        fprintf('测试 N = %d, M = %d\n', N, M);
        
        % 生成随机转移概率矩阵 P(Y|X)，确保每一行和为1
        P_Y_given_X = rand(N, M);
        P_Y_given_X = P_Y_given_X ./ sum(P_Y_given_X, 2);
        
        %% 1. BA算法
        try
            tic;
            C_BA = arimoto_blahut(P_Y_given_X); 
            time_BA(i,j) = toc;
        catch ME
            time_BA(i,j) = NaN;
            fprintf('BA算法优化失败: N=%d, M=%d\n', N, M);
        end
        
        %% 2. KKT条件法
         
            try
                tic;
                C_KKT = compute_capacity_KKT(P_Y_given_X);  % 假设 KKT_capacity 已定义
                time_KKT(i,j) = toc;
            catch ME
                time_KKT(i,j) = NaN;
                fprintf('KKT条件法优化失败: N=%d, M=%d\n', N, M);
            end
        
        
        %% 3. 引入惩罚项方法
        try
            tic;
            C_Penalty = penalty_capacity(P_Y_given_X, lambda, mu);  % 假设 penalty_capacity 已定义
            time_Penalty(i,j) = toc;
        catch ME
            time_Penalty(i,j) = NaN;
            fprintf('惩罚项方法优化失败: N=%d, M=%d\n', N, M);
        end
    end
end

total_time = toc(tStart);
fprintf('复杂度测试完成，总耗时: %.2f 秒\n', total_time);

%% 三、绘制复杂度增长曲线


% 1. BA算法的三维图
figure;
surf(input_symbols, output_symbols, time_BA', 'EdgeColor', 'none');
xlabel('输入符号数 N', 'FontSize', 12);
ylabel('输出符号数 M', 'FontSize', 12);
zlabel('运行时间 (秒)', 'FontSize', 12);
title('BA算法运行时间随 N 和 M 的变化', 'FontSize', 14);
colorbar;
grid on;
view(45, 30);

% 2. KKT条件法的三维图
figure;
surf(input_symbols, output_symbols, time_KKT', 'EdgeColor', 'none');
xlabel('输入符号数 N', 'FontSize', 12);
ylabel('输出符号数 M', 'FontSize', 12);
zlabel('运行时间 (秒)', 'FontSize', 12);
title('KKT条件法运行时间随 N 和 M 的变化', 'FontSize', 14);
colorbar;
grid on;
view(45, 30);

% 3. 引入惩罚项方法的三维图
figure;
surf(input_symbols, output_symbols, time_Penalty', 'EdgeColor', 'none');
xlabel('输入符号数 N', 'FontSize', 12);
ylabel('输出符号数 M', 'FontSize', 12);
zlabel('运行时间 (秒)', 'FontSize', 12);
title('引入惩罚项方法运行时间随 N 和 M 的变化', 'FontSize', 14);
colorbar;
grid on;
view(45, 30);


% 引入惩罚项方法函数
function C = penalty_capacity(P_Y_given_X, lambda, mu)
    [N, M] = size(P_Y_given_X);
    % 定义目标函数
    objective = @(P_X) -compute_mutual_information(P_X, P_Y_given_X) ...
                      + lambda * (sum(P_X) - 1)^2 ...
                      + mu * sum(max(0, -P_X).^2);
    % 初始猜测
    initial_P_X = ones(N,1)/N;
    % 优化
    options = optimoptions('fminunc', 'Display', 'off', 'Algorithm', 'quasi-newton', ...
                           'MaxIterations', 1000, 'OptimalityTolerance', 1e-12, ...
                           'StepTolerance', 1e-12);
    [P_X_opt, C_neg] = fminunc(objective, initial_P_X, options);
    % 修正
    P_X_opt = max(P_X_opt, 0);
    P_X_opt = P_X_opt / sum(P_X_opt);
    % 计算互信息
    C = compute_mutual_information(P_X_opt, P_Y_given_X);
end

% 互信息计算函数
function IXY = compute_mutual_information(P_X, P_Y_given_X)
    P_Y = P_X' * P_Y_given_X; % 1xM
    P_Y = P_Y + 1e-12; % 避免log2(0)
    H_Y = -sum(P_Y .* log2(P_Y));
    H_Y_given_X = 0;
    for x = 1:length(P_X)
        H_Y_given_X = H_Y_given_X - P_X(x) * sum(P_Y_given_X(x, :) .* log2(P_Y_given_X(x, :) + 1e-12));
    end
    IXY = H_Y - H_Y_given_X;
end

