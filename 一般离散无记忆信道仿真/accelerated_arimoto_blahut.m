%通过自适应步长优化传统BA算法

function [C, optimal_p, iter] = accelerated_arimoto_blahut(W, tol, max_iter)
% ACCELERATED_ARIMOTO_BLAHUT_ADAPTIVE 计算离散无记忆信道的容量及最佳输入分布（改进版）
%
% 输入参数:
%   W        - 信道转移概率矩阵 (输入符号数 x 输出符号数)，每行之和为1
%   tol      - 收敛阈值（默认 1e-6）
%   max_iter - 最大迭代次数（默认 1000）
%
% 输出参数:
%   C        - 信道容量 (bits/符号)
%   optimal_p- 最佳输入分布 (列向量)
%   iter     - 实际迭代次数

    % 参数默认值设置
    if nargin < 2
        tol = 1e-6;
    end
    if nargin < 3
        max_iter = 1000;
    end

    [num_x, num_y] = size(W);

    % 检查信道矩阵
    if any(W(:) < 0) || any(abs(sum(W, 2) - 1) > 1e-12)
        error('信道矩阵 W 必须是每行元素非负且每行和为1的概率矩阵。');
    end

    % 初始化输入分布为均匀分布
    p = ones(num_x, 1) / num_x;
    p_prev = p;

    C_prev = 0;
    iter = 0;

    % 定义 KLD 计算函数
    function d = kld(p1, p2)
        p1 = p1 + 1e-12; % 避免除以零
        p2 = p2 + 1e-12;
        d = sum(p1 .* log2(p1 ./ p2));
    end

    while iter < max_iter
        iter = iter + 1;

        % 计算输出分布
        py = W' * p;
        py(py == 0) = 1e-12;

        % 计算 log2(W(y|x)/py(y))
        log_ratio = log2(W ./ py');
        log_ratio(W == 0) = 0;

        % 计算互信息
        I = sum(p .* sum(W .* log_ratio, 2));

        % 检查收敛
        if abs(I - C_prev) < tol
            break;
        end

        % 计算辅助变量 q(x)
        q = sum(W .* log_ratio, 2);

        % 计算自适应步长 mu_k
        if iter > 1
            d_output = kld(py, py_prev); % D(Q p^k || Q p^{k-1})
            d_input = kld(p, p_prev);   % D(p^k || p^{k-1})
            if d_input > 1e-5  % 阈值，避免分母过小
                mu = d_output / d_input;
                mu = min(mu, 10); % 设置上限
            else
                mu = 1; % 接近收敛时切换到传统 BA
            end
        else
            mu = 5; % 第一次迭代
        end

        % 加速更新
        exp_q = 2.^(mu * q);
        p_new = p .* exp_q;
        p_new = p_new / sum(p_new);

        % 更新变量
        p_prev = p;
        py_prev = py;
        p = p_new;
        C_prev = I;
    end

    C = C_prev;
    optimal_p = p;

    if iter == max_iter
        warning('算法未在最大迭代次数内收敛。');
    end
end
