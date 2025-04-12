function [C, optimal_p, iter] = arimoto_blahut_natural_gradient(W, tol, max_iter)
% NATURAL_GRADIENT_BA_ADAPTIVE 使用自然梯度和自适应步长计算信道容量
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

    C_prev = 0;
    iter = 0;

    % 自适应步长参数
    mu_0 = 3;  % 初始步长
    alpha = 0.1; % 衰减率

    while iter < max_iter
        iter = iter + 1;

        % 计算输出分布 p(y) = sum_x p(x) W(x,y)
        py = W' * p;
        py(py == 0) = 1e-12;

        % 计算 log2(W(y|x)/py(y))
        log_ratio = log2(W ./ py');
        log_ratio(W == 0) = 0;

        % 计算互信息 I(X; Y)
        I = sum(p .* sum(W .* log_ratio, 2));

        % 检查收敛
        if abs(I - C_prev) < tol
            break;
        end

        % 计算辅助变量 q(x) = D(Q_j || Q p^k)
        q = sum(W .* log_ratio, 2);

        % 计算自适应步长 mu_k
        mu_k = max(mu_0 / (1 + alpha * (iter - 1)), 1);

        % 自然梯度更新
        p_new = p .* (1 + mu_k * (q - I));
        p_new = max(p_new, 0); % 确保非负性
        p_new = p_new / sum(p_new); % 归一化

        % 更新变量
        p = p_new;
        C_prev = I;
    end

    C = C_prev;
    optimal_p = p;

    if iter == max_iter
        warning('算法未在最大迭代次数内收敛。');
    end
end