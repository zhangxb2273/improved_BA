function trained_model = train_ba_initialization_model(training_data)
% TRAIN_BA_INITIALIZATION_MODEL 训练神经网络预测BA算法的初始输入分布
%
% 该函数训练一个神经网络模型，基于信道转移矩阵预测Blahut-Arimoto算法的优化初始输入分布
%
% 输入参数:
%   training_data - 包含W矩阵和最优p分布的训练数据集合
%
% 输出参数:
%   trained_model - 训练好的神经网络模型

    % 计算训练数据中最大的矩阵维度
    max_input_size = 0;   % 最大输入符号数
    max_output_size = 0;  % 最大输出符号数


    for i = 1:length(training_data)
        [rows, cols] = size(training_data{i}.W);  % 获取每个信道矩阵的尺寸
        max_input_size = max(max_input_size, rows);  % 更新最大输入符号数
        max_output_size = max(max_output_size, cols);  % 更新最大输出符号数
    end
    
    % 计算展平后的输入大小
    flat_input_size = max_input_size * max_output_size;
    
    % 创建神经网络层
    layers = create_ba_initialization_network(flat_input_size, max_input_size);
    
    % 准备训练数据
    numSamples = length(training_data);  % 训练样本数量
    
    % 预分配矩阵以提高效率
    X = zeros(numSamples, flat_input_size);  % 输入数据矩阵
    Y = zeros(numSamples, max_input_size);   % 目标输出矩阵
    
    for i = 1:numSamples
        W = training_data{i}.W;         % 获取信道矩阵
        p_opt = training_data{i}.optimal_p;  % 获取最优输入分布
        
        % 展平并填充W矩阵
        W_flat = W(:)';  % 展平为行向量
        if length(W_flat) < flat_input_size
            W_flat = [W_flat, zeros(1, flat_input_size - length(W_flat))];  % 零填充
        else
            W_flat = W_flat(1:flat_input_size);  % 截断
        end
        
        % 确保p_opt是列向量
        if size(p_opt, 2) > 1 && size(p_opt, 1) == 1
            p_opt = p_opt';  % 转置为列向量
        end
        
        % 填充p_opt向量
        if length(p_opt) < max_input_size
            p_padded = [p_opt; zeros(max_input_size - length(p_opt), 1)];  % 零填充
        else
            p_padded = p_opt(1:max_input_size);  % 截断
        end
        
        X(i,:) = W_flat;           % 存储输入数据
        Y(i,:) = p_padded';        % 存储目标输出（转置为行向量）
    end
    
    
    % 设置训练选项
    options = trainingOptions('adam', ...
        'MaxEpochs', 1000, ...                     % 增加到1000轮
        'MiniBatchSize', 16, ...
        'InitialLearnRate', 0.001, ...
        'LearnRateSchedule', 'piecewise', ...      % 学习率衰减策略
        'LearnRateDropFactor', 0.1, ...            % 学习率衰减因子
        'LearnRateDropPeriod', 200, ...            % 每200轮衰减一次学习率
        'L2Regularization', 0.0001, ...            % 添加L2正则化
        'ValidationFrequency', 50, ...             % 每50次迭代验证一次
        'ValidationPatience', 50, ...              % 提前停止耐心值
        'Shuffle', 'every-epoch', ...
        'Verbose', true, ...
        'Plots', 'training-progress', ...
        'CheckpointPath', 'checkpoints', ...       % 添加检查点保存路径
        'CheckpointFrequency', 100);               % 每100轮保存一次检查点
    
    
    % 训练神经网络
    trained_model = trainNetwork(X, Y, layers, options);
    
    fprintf('训练完成！\n');  % 输出训练完成消息
end
