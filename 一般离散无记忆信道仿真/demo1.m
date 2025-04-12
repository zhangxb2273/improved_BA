clear; clc;

% 定义信道条件概率矩阵 P(Y|X)
P_Y_given_X = [
    0.6,0.3,0.1;
    0.1,0.7,0.2
];

% 设置惩罚参数
lambda = 100; % 归一化约束的惩罚系数
mu = 1000;    % 非负约束的惩罚系数

%% 方法1：基于惩罚的优化方法
fprintf('==================== 方法1：基于惩罚的优化方法 ====================\n');

% 计算信道容量和最优输入分布
[P_X_opt_penalty, C_penalty] = compute_capacity_penalty(P_Y_given_X, lambda, mu);

% 显示结果
fprintf('最佳输入分布 p: [');
fprintf('%.4f ', P_X_opt_penalty);
fprintf(']\n');
fprintf('信道容量 C: %.4f 比特/符号\n', C_penalty);

fprintf('-----------------------------------------------------------------------\n\n');

%% 方法2：Arimoto-Blahut 算法
fprintf('==================== 方法2：Arimoto-Blahut 算法 =====================\n');

% 调用 Arimoto-Blahut 算法
[C_arimoto, optimal_p_arimoto, iterations_arimoto] = arimoto_blahut(P_Y_given_X);

% 显示结果
fprintf('信道容量 C: %.4f 比特/符号\n', C_arimoto);
fprintf('最佳输入分布 p: [');
fprintf('%.4f ', optimal_p_arimoto);
fprintf(']\n');
fprintf('迭代次数: %d 次\n', iterations_arimoto);

fprintf('-----------------------------------------------------------------------\n\n');

%% 方法3：KKT 条件法
fprintf('==================== 方法3：KKT 条件法 =============================\n');

% 计算信道容量和最优输入分布
[P_X_opt_KKT, C_KKT] = compute_capacity_KKT(P_Y_given_X);

% 显示结果
fprintf('最优输入分布 P_X: [');
fprintf('%.4f ', P_X_opt_KKT);
fprintf(']\n');
fprintf('信道容量 C: %.4f 比特/符号\n', C_KKT);

fprintf('-----------------------------------------------------------------------\n');
