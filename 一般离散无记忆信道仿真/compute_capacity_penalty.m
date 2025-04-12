%通过映入惩罚项求信道容量

function [P_X_opt, C] = compute_capacity_penalty(P_Y_given_X, lambda, mu)
    % compute_capacity_penalty 计算一般离散无记忆信道的信道容量和最优输入分布
    % 输入:
    %   P_Y_given_X - 转移概率矩阵，行表示输入符号，列表示输出符号，行和为1
    %   lambda - 归一化约束的惩罚系数
    %   mu - 非负约束的惩罚系数
    % 输出:
    %   P_X_opt - 最优输入分布向量
    %   C - 信道容量（比特/符号）

    [num_inputs, num_outputs] = size(P_Y_given_X);
    if any(abs(sum(P_Y_given_X, 2) - 1) > 1e-6)
        error('转移概率矩阵的每一行之和必须为1。');
    end

    % 定义目标函数，包括惩罚项
    objective = @(P_X) -compute_mutual_information(P_X, P_Y_given_X) ...
                      + lambda * (sum(P_X) - 1)^2 ...
                      + mu * sum(max(0, -P_X).^2);

    % 初始猜测: 均匀分布
    initial_P_X = ones(num_inputs, 1) / num_inputs;

    % 设置优化选项
    options = optimoptions('fminunc', 'Display', 'iter', 'Algorithm', 'quasi-newton', ...
                           'MaxIterations', 1000, 'OptimalityTolerance', 1e-12, 'StepTolerance', 1e-12);

    % 使用fminunc进行优化
    [P_X_opt, neg_C] = fminunc(objective, initial_P_X, options);

    % 修正微小的负值并归一化
    P_X_opt = max(P_X_opt, 0);
    P_X_opt = P_X_opt / sum(P_X_opt);

    % 计算信道容量
    C = compute_mutual_information(P_X_opt, P_Y_given_X);
end

function IXY = compute_mutual_information(P_X, P_Y_given_X)
    % compute_mutual_information 计算互信息 I(X; Y)
    % 输入:
    %   P_X - 输入分布向量
    %   P_Y_given_X - 转移概率矩阵
    % 输出:
    %   IXY - 互信息（比特）

    P_Y = P_X' * P_Y_given_X; % 1xY vector
    P_Y = P_Y + 1e-12; % 避免log2(0)
    H_Y = -sum(P_Y .* log2(P_Y));

    % 条件熵 H(Y|X)
    H_Y_given_X = 0;
    for x = 1:length(P_X)
        H_Y_given_X = H_Y_given_X - P_X(x) * sum(P_Y_given_X(x, :) .* log2(P_Y_given_X(x, :) + 1e-12));
    end

    IXY = H_Y - H_Y_given_X;
end
