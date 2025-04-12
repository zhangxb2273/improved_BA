%通过KKT方法算信道容量

function [P_X_opt, C] = compute_capacity_KKT(P_Y_given_X)
    % compute_capacity_numerical 通过数值优化计算离散无记忆信道的信道容量和最优输入分布
    % 输入:
    %   P_Y_given_X - 转移概率矩阵，行表示输入符号，列表示输出符号，行和为1
    % 输出:
    %   P_X_opt - 最优输入分布向量
    %   C - 信道容量（比特/符号）
    
    num_inputs = size(P_Y_given_X, 1);
    
    % 初始猜测：均匀分布
    P0 = ones(num_inputs, 1) / num_inputs;
    
    % 目标函数：负互信息（因为 fmincon 是最小化函数）
    function neg_I = objective(P)
        P_Y = P' * P_Y_given_X;
        P_Y = P_Y + 1e-12; % 避免 log2(0)
        H_Y = -sum(P_Y .* log2(P_Y));
        H_Y_given_X = -sum(P .* sum(P_Y_given_X .* log2(P_Y_given_X + 1e-12), 2));
        IXY = H_Y - H_Y_given_X;
        neg_I = -IXY;
    end

    % 约束条件
    A = []; b = [];
    Aeq = ones(1, num_inputs);
    beq = 1;
    lb = zeros(num_inputs, 1);
    ub = ones(num_inputs, 1);
    
    % 优化选项
    options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp');
    
    % 求解优化问题
    [P_X_opt, neg_C] = fmincon(@objective, P0, A, b, Aeq, beq, lb, ub, [], options);
    
    C = -neg_C; % 互信息
end
