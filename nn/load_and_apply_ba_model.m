function [C, optimal_p, iter, time_used] = load_and_apply_ba_model(W, model_filename, tol, max_iter)
% LOAD_AND_APPLY_BA_MODEL 加载训练好的模型并应用于新的信道矩阵
%
% 该函数加载训练好的神经网络模型，然后使用它优化BA算法的初始分布
%
% 输入参数:
%   W             - 信道转移概率矩阵
%   model_filename- 保存模型的文件名
%   tol           - 收敛阈值 (默认 1e-6)
%   max_iter      - 最大迭代次数 (默认 1000)
%
% 输出参数:
%   C             - 信道容量 (bits/符号)
%   optimal_p     - 最佳输入分布 (列向量)
%   iter          - 实际迭代次数
%   time_used     - 运行时间(秒)

    % 参数默认值设置
    if nargin < 3
        tol = 1e-6;  % 默认收敛阈值
    end
    if nargin < 4
        max_iter = 1000;  % 默认最大迭代次数
    end

    % 加载模型
    fprintf('正在加载模型: %s\n', model_filename);
    loaded_data = load(model_filename);
    model = loaded_data.model;
    
    % 运行神经网络优化的BA算法
    tic;  % 开始计时
    [C, optimal_p, iter, p_initial] = neural_arimoto_blahut(W, model, tol, max_iter);
    time_used = toc;  % 结束计时
    
    % 显示结果
    fprintf('信道容量: %.6f bits/符号\n', C);
    fprintf('迭代次数: %d\n', iter);
    fprintf('运行时间: %.4f 秒\n', time_used);
end
