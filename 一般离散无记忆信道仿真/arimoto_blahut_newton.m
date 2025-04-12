%通过牛顿法优化传统BA算法

function [C, optimal_p, iter] = arimoto_blahut_newton(W, tol, max_iter)
    % 参数默认值设置
    if nargin < 2, tol = 1e-6; end
    if nargin < 3, max_iter = 1000; end

    [num_x, ~] = size(W);
    % 检查信道矩阵
    if any(W(:) < 0) || any(abs(sum(W, 2) - 1) > 1e-12)
        error('信道矩阵 W 必须是每行元素非负且每行和为1的概率矩阵。');
    end

    % 定义目标函数和梯度
    function [I, grad_z] = objective(z)
        p = exp(z) / sum(exp(z)); % softmax
        py = W' * p; % 输出分布
        py(py == 0) = 1e-12; % 避免除零
        log_ratio = log2(W ./ py'); % W(x,y)/p(y)
        log_ratio(W == 0) = 0; % 处理W=0
        I = sum(p .* sum(W .* log_ratio, 2)); % 互信息
        % 计算梯度
        q = sum(W .* log_ratio, 2); % ∂I/∂p(x)
        grad_z = p .* (q - I); % ∂I/∂z(a)
        I = -I; % fminunc求最小化，目标取负值
        grad_z = -grad_z;
    end

    % 初始值
    z0 = zeros(num_x, 1); % 对应均匀分布
    options = optimoptions('fminunc', ...
        'Algorithm', 'quasi-newton', ... % 使用BFGS
        'SpecifyObjectiveGradient', true, ...
        'TolFun', tol, ...
        'MaxIterations', max_iter, ...
        'Display', 'off');

    % 优化
    [z_opt, fval, exitflag, output] = fminunc(@objective, z0, options);
    optimal_p = exp(z_opt) / sum(exp(z_opt)); % 最佳输入分布
    C = -fval; % 信道容量
    iter = output.iterations;

    if exitflag <= 0
        warning('优化未正常收敛。');
    end
end
