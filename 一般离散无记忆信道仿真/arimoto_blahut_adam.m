%通过引入adam自适应步长优化传统BA算法

function [C, optimal_p, iter] = arimoto_blahut_adam(W, tol, max_iter, alpha, beta1, beta2, epsilon)
% ARIMOTO_BLAHUT_ADAM 使用Adam优化器的Blahut-Arimoto算法
%
% 输入参数:
%   W        - 信道转移概率矩阵 (输入符号数 x 输出符号数)，每行之和为1
%   tol      - 收敛阈值 (默认 1e-6)
%   max_iter - 最大迭代次数 (默认 1000)
%   alpha    - 学习率 (默认 0.01)
%   beta1    - 一阶矩衰减率 (默认 0.9)
%   beta2    - 二阶矩衰减率 (默认 0.999)
%   epsilon  - 避免除零的小正数 (默认 1e-8)
%
% 输出参数:
%   C        - 信道容量 (bits/符号)
%   optimal_p- 最佳输入分布 (列向量)
%   iter     - 实际迭代次数

    % 参数默认值设置
    if nargin < 2, tol = 1e-6; end
    if nargin < 3, max_iter = 1000; end
    if nargin < 4, alpha = 0.1; end
    if nargin < 5, beta1 = 0.9; end
    if nargin < 6, beta2 = 0.999; end
    if nargin < 7, epsilon = 1e-8; end

    [num_x, num_y] = size(W); % 输入符号数 x 输出符号数

    % 检查信道矩阵是否为有效的概率矩阵
    if any(W(:) < 0) || any(abs(sum(W, 2) - 1) > 1e-12)
        error('信道矩阵 W 必须是每行元素非负且每行和为1的概率矩阵。');
    end

    % 初始化 z 为零，对应均匀分布 p(x) = 1/num_x
    z = zeros(num_x, 1);
    p = ones(num_x, 1) / num_x;

    % 初始化Adam参数
    m = zeros(num_x, 1); % 一阶矩
    v = zeros(num_x, 1); % 二阶矩
    t = 0; % 迭代计数器

    C_prev = 0;
    iter = 0;

    while iter < max_iter
        iter = iter + 1;
        t = t + 1;

        % 计算输出分布 p(y) = sum_x p(x) W(x,y)
        py = W' * p;
        py(py == 0) = 1e-12; % 避免除零

        % 计算 log2(W(x,y)/p(y))
        log_ratio = log2(W ./ py');
        log_ratio(W == 0) = 0; % 处理 W(x,y)=0 的情况

        % 计算 q(x) = sum_y W(x,y) * log2(W(x,y)/p(y))
        q = sum(W .* log_ratio, 2);

        % 计算互信息 I(X;Y) = sum_x p(x) * q(x)
        I = sum(p .* q);

        % 计算梯度 g(x) = p(x) * (q(x) - I)
        g = p .* (q - I);

        % Adam更新
        m = beta1 * m + (1 - beta1) * g; % 更新一阶矩
        v = beta2 * v + (1 - beta2) * (g .^ 2); % 更新二阶矩
        m_hat = m / (1 - beta1^t); % 偏差校正
        v_hat = v / (1 - beta2^t);
        z = z + alpha * m_hat ./ (sqrt(v_hat) + epsilon); % 更新 z

        % 更新输入分布 p(x)
        p = exp(z) / sum(exp(z));

        % 检查收敛
        if abs(I - C_prev) < tol
            break;
        end
        C_prev = I;
    end

    % 最终信道容量
    C = C_prev;
    optimal_p = p;

    % 如果达到最大迭代次数，给出警告
    if iter == max_iter
        warning('算法未在最大迭代次数内收敛。');
    end
end
