function [C, optimal_p, iter, p_initial] = neural_arimoto_blahut(W, model, tol, max_iter)
% NEURAL_ARIMOTO_BLAHUT 使用神经网络优化的Blahut-Arimoto算法
%
% 该函数使用神经网络预测更好的初始输入分布，从而加速Blahut-Arimoto算法的收敛过程
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
        tol = 1e-6;  % 默认收敛阈值
    end
    if nargin < 4
        max_iter = 1000;  % 默认最大迭代次数
    end

    [num_x, num_y] = size(W);  % 获取输入符号数和输出符号数

    % 检查信道矩阵是否为有效的概率矩阵
    if any(W(:) < 0) || any(abs(sum(W, 2) - 1) > 1e-12)
        error('信道矩阵 W 必须是每行元素非负且每行和为1的概率矩阵。');
    end

    % 获取神经网络输入层的大小
    input_layer = model.Layers(1);
    input_size = input_layer.InputSize;  % 输入层大小
    
    % 展平信道矩阵
    W_flat = W(:)';  % 将W矩阵展平为行向量
    
    % 如果需要，填充或截断输入数据
    if length(W_flat) < input_size
        W_flat = [W_flat, zeros(1, input_size - length(W_flat))];  % 零填充
    else
        W_flat = W_flat(1:input_size);  % 截断
    end
    
    % 使用神经网络预测初始输入分布
    p_pred = predict(model, W_flat);  % 预测分布
    
    % 确保只取需要的部分
    p_pred = p_pred(1, 1:min(num_x, size(p_pred, 2)));  % 获取正确的子集
    
    % 如果预测结果不够长，则填充
    if length(p_pred) < num_x
        p_pred = [p_pred, zeros(1, num_x - length(p_pred))];
    end
    
    % 确保分布非负
    p_pred = max(p_pred, 0);  % 将负值设为0
    
    % 归一化分布
    if sum(p_pred) > 0
        p_initial = p_pred' / sum(p_pred);  % 转置为列向量并归一化
    else
        % 如果预测全为零，则使用均匀分布
        p_initial = ones(num_x, 1) / num_x;
    end
    
    % 使用预测的分布作为初始分布
    p = p_initial;

    C_prev = 0;  % 上一次迭代的容量值
    iter = 0;    % 迭代计数器

    % 主迭代循环
    while iter < max_iter
        iter = iter + 1;  % 增加迭代计数

        % 计算输出分布 p(y) = sum_x p(x) W(x,y)
        py = W' * p;  % (num_y x 1) 输出分布向量

        % 避免除以零，添加一个极小值
        py(py < 1e-12) = 1e-12;  % 对接近零的概率进行修正

        % 计算 log2(W(y|x)/py(y))
        log_ratio = log2(W ./ py');  % (num_x x num_y) 对数比率矩阵

        % 处理 W(x,y) = 0 的情况，避免 NaN
        log_ratio(W < 1e-12) = 0;  % 当条件概率为0时，对数比率也设为0

        % 计算互信息 I(X; Y)
        I = sum(p .* sum(W .* log_ratio, 2));  % 当前互信息（信道容量估计）

        % 检查收敛
        if abs(I - C_prev) < tol
            break;  % 如果改进小于阈值，则结束迭代
        end

        % 计算辅助变量 q(x)
        q = sum(W .* log_ratio, 2);  % (num_x x 1) 辅助向量

        % 更新输入分布
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
end
