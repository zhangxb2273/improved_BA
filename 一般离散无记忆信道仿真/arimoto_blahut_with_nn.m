function [C, optimal_p, iter, p_initial] = neural_arimoto_blahut(W, model, tol, max_iter)
% NEURAL_ARIMOTO_BLAHUT 使用神经网络优化的BA算法
%
% 输入参数:
%   W        - 信道转移概率矩阵 (输入符号数 x 输出符号数)，每行之和为1
%   model    - 训练好的神经网络模型
%   tol      - 收敛阈值 (默认 1e-6)
%   max_iter - 最大迭代次数 (默认 1000)
%
% 输出参数:
%   C        - 信道容量 (bits/符号)
%   optimal_p- 最佳输入分布 (列向量)
%   iter     - 实际迭代次数
%   p_initial- 神经网络预测的初始输入分布

    % 参数默认值设置
    if nargin < 3
        tol = 1e-6;
    end
    if nargin < 4
        max_iter = 1000;
    end

    [num_x, num_y] = size(W); % 输入符号数 x 输出符号数

    % 检查信道矩阵是否为有效的概率矩阵
    if any(W(:) < 0) || any(abs(sum(W, 2) - 1) > 1e-12)
        error('信道矩阵 W 必须是每行元素非负且每行和为1的概率矩阵。');
    end

    % 使用神经网络预测初始输入分布
    W_flat = W(:)'; % 展平并转置为行向量
    
    % 调整大小以适配模型输入
    max_size = sqrt(length(model.Layers(1).InputSize));
    if length(W_flat) < model.Layers(1).InputSize
        W_flat = [W_flat, zeros(1, model.Layers(1).InputSize - length(W_flat))];
    else
        W_flat = W_flat(1:model.Layers(1).InputSize);
    end
    
    % 预测初始输入分布
    p_pred = predict(model, W_flat);
    p_pred = p_pred(1:num_x);
    p_initial = p_pred / sum(p_pred); % 确保归一化
    
    % 使用预测的分布作为初始分布
    p = p_initial;

    C_prev = 0;
    iter = 0;

    while iter < max_iter
        iter = iter + 1;

        % 计算输出分布 p(y) = sum_x p(x) W(x,y)
        py = W' * p; % (num_y x 1)

        % 避免除以零，添加一个极小值
        py(py == 0) = 1e-12;

        % 计算 log2(W(y|x)/py(y))，其中 W 为 (num_x x num_y)
        log_ratio = log2(W ./ py'); % (num_x x num_y)

        % 处理 W(x,y) = 0 的情况，避免 NaN
        log_ratio(W == 0) = 0;

        % 计算互信息 I(X; Y) = sum_x p(x) sum_y W(x,y) log2(W(x,y)/p(y))
        I = sum(p .* sum(W .* log_ratio, 2)); % 标量

        % 检查收敛
        if abs(I - C_prev) < tol
            break;
        end

        % 计算辅助变量 q(x) = sum_y W(x,y) log2(W(x,y)/p(y))
        q = sum(W .* log_ratio, 2); % (num_x x 1)

        % 更新输入分布 p(x) = p(x) * 2^(q(x)) / Z，其中 Z = sum_x p(x) * 2^(q(x))
        exp_q = 2.^q;
        p_new = p .* exp_q;
        p_new = p_new / sum(p_new); % 归一化

        % 更新变量
        p = p_new;
        C_prev = I;
    end

    % 最终信道容量
    C = C_prev;

    % 最佳输入分布
    optimal_p = p;
end