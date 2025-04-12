%求最佳超参数

% 清除工作区并关闭命令行窗口
clear; clc;

% 生成50×50的信道转移概率矩阵W
num_x = 50;
num_y = 50;
W = rand(num_x, num_y);         % 生成随机矩阵
W = W ./ sum(W, 2);             % 归一化，每行和为1

% 设置收敛参数
tol = 1e-6;                     % 收敛阈值
max_iter = 1000;                % 最大迭代次数

% 定义超参数取值范围
mu_0_values = 10:1:100;           % mu_0从1到10，步长1
alpha_values = 0.01:0.01:0.1;     % alpha从0.1到2.0，步长0.1

% 初始化时间矩阵
time_matrix = zeros(length(mu_0_values), length(alpha_values));

% 设置重复运行次数以减少随机性
num_runs = 5;

% 网格搜索：测试每组mu_0和alpha
for i = 1:length(mu_0_values)
    mu_0 = mu_0_values(i);
    for j = 1:length(alpha_values)
        alpha = alpha_values(j);
        total_time = 0;
        % 多次运行取平均时间
        for k = 1:num_runs
            tic;                        % 开始计时
            [C, optimal_p, iter] = natural_gradient_ba(W, tol, max_iter, mu_0, alpha);
            time = toc;                 % 结束计时
            total_time = total_time + time;
        end
        average_time = total_time / num_runs;
        time_matrix(i,j) = average_time;
        fprintf('mu_0 = %d, alpha = %.1f, 平均时间 = %.4f 秒\n', mu_0, alpha, average_time);
    end
end

% 找到最短时间对应的超参数
[min_time, idx] = min(time_matrix(:));
[i, j] = ind2sub(size(time_matrix), idx);
best_mu_0 = mu_0_values(i);
best_alpha = alpha_values(j);

% 输出最佳超参数
fprintf('最佳 mu_0: %d\n', best_mu_0);
fprintf('最佳 alpha: %.1f\n', best_alpha);
fprintf('最小平均时间: %.4f 秒\n', min_time);

% 可视化结果：绘制热图
figure;
imagesc(alpha_values, mu_0_values, time_matrix);
colorbar;                       % 添加颜色条
xlabel('alpha');
ylabel('mu_0');
title('平均计算时间 (秒)');
set(gca, 'YDir', 'normal');     % 使mu_0从下到上递增
