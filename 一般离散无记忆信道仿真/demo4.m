% complexity_analysis_independent.m
% 该脚本分析 BA 算法及其变体在不同信道维数下的计算复杂度和精确度。
% 独立变化输入符号数 N 和输出符号数 M，并绘制三维图表。

clear; clc; close all;
P = [0.5, 0.5; 0.5, 0.5];

% 先调用一次函数，防止污染第一个数据
BA_test = arimoto_blahut(P); 
Adam_test = arimoto_blahut_adam(P);
AccBA_test = accelerated_arimoto_blahut(P); 
newton_test = arimoto_blahut_newton(P);

%% 一、定义参数范围
input_symbols = 2:5:102;  % 输入符号数 N 从 2 到 102，每次增加 5
output_symbols = 2:5:102; % 输出符号数 M 从 2 到 102，每次增加 5
num_N = length(input_symbols);
num_M = length(output_symbols);

% 初始化运行时间和相对误差存储矩阵
time_adam = zeros(num_N, num_M);
time_BA = zeros(num_N, num_M);
time_newton = zeros(num_N, num_M); 
time_NG = zeros(num_N, num_M);
error_adam = zeros(num_N, num_M); % 相对误差矩阵
error_BA = zeros(num_N, num_M);
error_newton = zeros(num_N, num_M);
error_NG = zeros(num_N, num_M);

%% 二、进行复杂度和精确度测试
fprintf('开始测试...\n');
tStart = tic;

for i = 1:num_N
    N = input_symbols(i);
    for j = 1:num_M
        M = output_symbols(j);
        fprintf('测试 N = %d, M = %d\n', N, M);
        
        % 生成随机转移概率矩阵 P(Y|X)，确保每一行和为 1
        P_Y_given_X = rand(N, M);
        P_Y_given_X = P_Y_given_X ./ sum(P_Y_given_X, 2);
        
        % 计算基准信道容量，使用高迭代次数的 BA 算法
        try
            C_benchmark = arimoto_blahut(P_Y_given_X, 1e-8, 10000); % 高精度设置
        catch
            C_benchmark = NaN;
        end
        
        % 检查基准值是否有效
        if isnan(C_benchmark) || C_benchmark < 1e-6
            time_BA(i,j) = NaN;
            time_newton(i,j) = NaN;
            time_NG(i,j) = NaN;
            time_adam(i,j) = NaN;
            error_BA(i,j) = NaN;
            error_newton(i,j) = NaN;
            error_NG(i,j) = NaN;
            error_adam(i,j) = NaN;
            continue;
        end
        
        %% 1. BA 算法
        try
            tic;
            C_BA = arimoto_blahut(P_Y_given_X); 
            time_BA(i,j) = toc;
            error_BA(i,j) = abs(C_BA - C_benchmark) / C_benchmark;
        catch
            time_BA(i,j) = NaN;
            error_BA(i,j) = NaN;
            fprintf('BA 算法优化失败: N=%d, M=%d\n', N, M);
        end
        
        %% 2. Newton 算法
        try
            tic;
            C_newton = arimoto_blahut_newton(P_Y_given_X); 
            time_newton(i,j) = toc;
            error_newton(i,j) = abs(C_newton - C_benchmark) / C_benchmark;
        catch
            time_newton(i,j) = NaN;
            error_newton(i,j) = NaN;
            fprintf('Newton 算法优化失败: N=%d, M=%d\n', N, M);
        end
        
        %% 3. Natural Gradient (NGBA) 算法
        try
            tic;
            C_NG = arimoto_blahut_natural_gradient(P_Y_given_X); 
            time_NG(i,j) = toc;
            error_NG(i,j) = abs(C_NG - C_benchmark) / C_benchmark;
        catch
            time_NG(i,j) = NaN;
            error_NG(i,j) = NaN;
            fprintf('NGBA 算法优化失败: N=%d, M=%d\n', N, M);
        end
        
        %% 4. Adam 算法
        try
            tic;
            C_adam = arimoto_blahut_adam(P_Y_given_X); 
            time_adam(i,j) = toc;
            error_adam(i,j) = abs(C_adam - C_benchmark) / C_benchmark;
        catch
            time_adam(i,j) = NaN;
            error_adam(i,j) = NaN;
            fprintf('Adam 算法优化失败: N=%d, M=%d\n', N, M);
        end
    end
end

total_time = toc(tStart);
fprintf('测试完成，总耗时: %.2f 秒\n', total_time);

%% 三、绘制复杂度增长曲线

% 1. BA 算法运行时间三维图
figure;
surf(input_symbols, output_symbols, time_BA', 'EdgeColor', 'none');
xlabel('输入符号数 N', 'FontSize', 12);
ylabel('输出符号数 M', 'FontSize', 12);
zlabel('运行时间 (秒)', 'FontSize', 12);
title('BA 算法运行时间随 N 和 M 的变化', 'FontSize', 14);
colorbar;
grid on;
view(45, 30);

% 2. Newton 算法运行时间三维图
figure;
surf(input_symbols, output_symbols, time_newton', 'EdgeColor', 'none');
xlabel('输入符号数 N', 'FontSize', 12);
ylabel('输出符号数 M', 'FontSize', 12);
zlabel('运行时间 (秒)', 'FontSize', 12);
title('Newton 算法运行时间随 N 和 M 的变化', 'FontSize', 14);
colorbar;
grid on;
view(45, 30);

% 3. NGBA 算法运行时间三维图
figure;
surf(input_symbols, output_symbols, time_NG', 'EdgeColor', 'none');
xlabel('输入符号数 N', 'FontSize', 12);
ylabel('输出符号数 M', 'FontSize', 12);
zlabel('运行时间 (秒)', 'FontSize', 12);
title('NGBA 算法运行时间随 N 和 M 的变化', 'FontSize', 14);
colorbar;
grid on;
view(45, 30);

% 4. Adam 算法运行时间三维图
figure;
surf(input_symbols, output_symbols, time_adam', 'EdgeColor', 'none');
xlabel('输入符号数 N', 'FontSize', 12);
ylabel('输出符号数 M', 'FontSize', 12);
zlabel('运行时间 (秒)', 'FontSize', 12);
title('Adam 算法运行时间随 N 和 M 的变化', 'FontSize', 14);
colorbar;
grid on;
view(45, 30);

%% 四、绘制精确度图表

% 1. BA 算法相对误差三维图
figure;
surf(input_symbols, output_symbols, error_BA', 'EdgeColor', 'none');
xlabel('输入符号数 N', 'FontSize', 12);
ylabel('输出符号数 M', 'FontSize', 12);
zlabel('相对误差', 'FontSize', 12);
title('BA 算法相对误差随 N 和 M 的变化', 'FontSize', 14);
colorbar;
grid on;
view(45, 30);

% 2. Newton 算法相对误差三维图
figure;
surf(input_symbols, output_symbols, error_newton', 'EdgeColor', 'none');
xlabel('输入符号数 N', 'FontSize', 12);
ylabel('输出符号数 M', 'FontSize', 12);
zlabel('相对误差', 'FontSize', 12);
title('Newton 算法相对误差随 N 和 M 的变化', 'FontSize', 14);
colorbar;
grid on;
view(45, 30);

% 3. NGBA 算法相对误差三维图
figure;
surf(input_symbols, output_symbols, error_NG', 'EdgeColor', 'none');
xlabel('输入符号数 N', 'FontSize', 12);
ylabel('输出符号数 M', 'FontSize', 12);
zlabel('相对误差', 'FontSize', 12);
title('NGBA 算法相对误差随 N 和 M 的变化', 'FontSize', 14);
colorbar;
grid on;
view(45, 30);

% 4. Adam 算法相对误差三维图
figure;
surf(input_symbols, output_symbols, error_adam', 'EdgeColor', 'none');
xlabel('输入符号数 N', 'FontSize', 12);
ylabel('输出符号数 M', 'FontSize', 12);
zlabel('相对误差', 'FontSize', 12);
title('Adam 算法相对误差随 N 和 M 的变化', 'FontSize', 14);
colorbar;
grid on;
view(45, 30);
