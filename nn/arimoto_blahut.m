function [C, optimal_p, iter] = arimoto_blahut(W, tol, max_iter)
% ARIMOTO_BLAHUT 计算离散无记忆信道的容量及最佳输入分布
%
% 该函数实现Blahut-Arimoto算法，迭代计算离散无记忆信道的容量和相应的最佳输入分布
%
% 输入参数:
%   W        - 信道转移概率矩阵 (输入符号数 x 输出符号数)，每行之和为1
%   tol      - 收敛阈值 (默认 1e-6)
%   max_iter - 最大迭代次数 (默认 2000)
%
% 输出参数:
%   C        - 信道容量 (bits/符号)
%   optimal_p- 最佳输入分布 (列向量)
%   iter     - 实际迭代次数

    % 参数默认值设置
    if nargin < 2
        tol = 1e-6;  % 默认收敛阈值
    end
    if nargin < 3
        max_iter = 2000;  % 默认最大迭代次数
    end

    [num_x, num_y] = size(W);  % 获取输入符号数和输出符号数

    % 检查信道矩阵是否为有效的概率矩阵
    if any(W(:) < 0) || any(abs(sum(W, 2) - 1) > 1e-12)
        error('信道矩阵 W 必须是每行元素非负且每行和为1的概率矩阵。');
    end

    % 初始化输入分布为均匀分布
    p = ones(num_x, 1) / num_x;

    C_prev = 0;  % 上一次迭代的容量值
    iter = 0;    % 迭代计数器
    
    % 用于提前停止的计数器
    stall_counter = 0;  % 跟踪连续小改进的次数
    stall_threshold = 5;  % 如果连续5次改善很小，则提前停止

    % 主迭代循环
    while iter < max_iter
        iter = iter + 1;  % 增加迭代计数

        % 计算输出分布 p(y) = sum_x p(x) W(x,y)
        py = W' * p;  % (num_y x 1) 输出分布向量

        % 避免除以零，添加一个极小值
        py(py < 1e-12) = 1e-12;  % 对接近零的概率进行修正

        % 计算 log2(W(y|x)/py(y))，其中 W 为 (num_x x num_y)
        log_ratio = log2(W ./ py');  % (num_x x num_y) 对数比率矩阵

        % 处理 W(x,y) = 0 的情况，避免 NaN
        log_ratio(W < 1e-12) = 0;  % 当条件概率为0时，对数比率也设为0

        % 计算互信息 I(X; Y) = sum_x p(x) sum_y W(x,y) log2(W(x,y)/p(y))
        I = sum(p .* sum(W .* log_ratio, 2));  % 当前互信息（信道容量估计）

        % 检查收敛
        improvement = abs(I - C_prev);  % 计算容量改进量
        if improvement < tol  % 如果改进小于阈值
            stall_counter = stall_counter + 1;  % 增加停滞计数
            if stall_counter >= stall_threshold  % 如果连续多次小改进
                break;  % 提前结束迭代
            end
        else
            stall_counter = 0;  % 重置停滞计数
        end

        % 计算辅助变量 q(x) = sum_y W(x,y) log2(W(x,y)/p(y))
        q = sum(W .* log_ratio, 2);  % (num_x x 1) 辅助向量

        % 更新输入分布 p(x) = p(x) * 2^(q(x)) / Z，其中 Z = sum_x p(x) * 2^(q(x))
        exp_q = 2.^q;  % 计算2的q次方
        p_new = p .* exp_q;  % 按元素相乘
        p_new = p_new / sum(p_new);  % 归一化新分布

        % 更新变量
        p = p_new;  % 更新输入分布
        C_prev = I;  % 更新前一次容量估计
    end

    % 最终信道容量
    C = C_prev;

    % 最佳输入分布
    optimal_p = p;

    % 如果达到最大迭代次数，但差距仍然很大，给出警告
    if iter == max_iter && stall_counter < stall_threshold
        warning('算法未在最大迭代次数内收敛。考虑增加max_iter或放宽tol。');
    end
end
