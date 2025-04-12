function demo_neural_ba()
% DEMO_NEURAL_BA 演示神经网络优化的Blahut-Arimoto算法
%
% 该函数演示完整的工作流程：生成训练数据，训练神经网络，
% 然后比较传统BA算法和神经网络优化的BA算法在测试案例上的表现

 % 创建检查点目录
    if ~exist('checkpoints', 'dir')
        mkdir('checkpoints');
    end
    
    % 生成更多训练数据以支持更长时间训练
    fprintf('正在生成训练数据...\n');
    training_data = generate_training_data(500, 8, 8);  % 增加样本数量
    
    % 划分训练集和验证集
    num_samples = length(training_data);
    shuffle_idx = randperm(num_samples);
    train_idx = shuffle_idx(1:round(0.8*num_samples));
    val_idx = shuffle_idx(round(0.8*num_samples)+1:end);
    
    training_data_train = training_data(train_idx);
    training_data_val = training_data(val_idx);
    
    % 训练神经网络
    fprintf('正在训练神经网络，设置为1000轮...\n');
    t_start = tic;
    model = train_ba_initialization_model(training_data_train);
    training_time = toc(t_start);
    fprintf('训练完成，用时 %.2f 分钟!\n', training_time/60);
    
    % 保存模型，添加更多信息
    model_filename = 'ba_neural_model_1000epochs.mat';
    fprintf('正在保存模型到文件: %s\n', model_filename);
    training_info.epochs = 1000;
    training_info.training_time = training_time;
    training_info.date = datestr(now);
    save(model_filename, 'model', 'training_info', 'training_data');
    fprintf('模型保存完成!\n\n');
    
    % 测试神经网络优化的BA算法
    fprintf('测试神经网络优化的BA算法...\n');
    
    % 生成测试数据
    test_cases = 5;  % 测试案例数量
    total_trad_iters = 0;  % 传统算法总迭代次数
    total_neural_iters = 0;  % 神经网络优化算法总迭代次数
    
    for i = 1:test_cases
        % 随机生成信道矩阵
        num_x = randi([2, 5]);  % 随机输入符号数
        num_y = randi([2, 5]);  % 随机输出符号数
        W = rand(num_x, num_y);  % 随机生成矩阵
        W = W ./ sum(W, 2);  % 每行归一化
        
        % 运行传统BA算法
        tic;  % 开始计时
        [C_trad, p_trad, iter_trad] = arimoto_blahut(W, 1e-6, 1000);  % 运行传统算法
        time_trad = toc;  % 结束计时
        
        % 运行神经网络优化的BA算法
        tic;  % 开始计时
        [C_neural, p_neural, iter_neural, p_initial] = neural_arimoto_blahut(W, model, 1e-6, 1000);  % 运行优化算法
        time_neural = toc;  % 结束计时
        
        % 累加迭代次数
        total_trad_iters = total_trad_iters + iter_trad;  % 累加传统算法迭代次数
        total_neural_iters = total_neural_iters + iter_neural;  % 累加优化算法迭代次数
        
        % 输出结果
        fprintf('测试 #%d (%dx%d信道):\n', i, num_x, num_y);
        fprintf('  传统BA:     容量=%.6f, 迭代次数=%d, 耗时=%.4f秒\n', C_trad, iter_trad, time_trad);
        fprintf('  神经网络BA: 容量=%.6f, 迭代次数=%d, 耗时=%.4f秒\n', C_neural, iter_neural, time_neural);
        fprintf('  迭代减少: %.2f%%\n\n', 100*(iter_trad-iter_neural)/iter_trad);
    end
    
    % 输出总结
    fprintf('总结:\n');
    fprintf('  传统BA平均迭代次数: %.2f\n', total_trad_iters/test_cases);
    fprintf('  神经网络BA平均迭代次数: %.2f\n', total_neural_iters/test_cases);
    fprintf('  平均迭代减少: %.2f%%\n', 100*(total_trad_iters-total_neural_iters)/total_trad_iters);
end
