function training_data = generate_training_data(num_samples, max_input_size, max_output_size)
% GENERATE_TRAINING_DATA 生成用于训练神经网络的信道数据集
%
% 该函数生成随机信道转移矩阵及其对应的最优输入分布，用于训练神经网络
%
% 输入参数:
%   num_samples    - 要生成的样本数量
%   max_input_size - 最大输入符号数
%   max_output_size- 最大输出符号数
%
% 输出参数:
%   training_data  - 包含W矩阵和最优p分布的训练数据

    training_data = cell(num_samples, 1);  % 初始化训练数据单元格数组
    
    % 显示进度信息
    fprintf('开始生成%d个训练样本...\n', num_samples);
    progress_step = max(1, floor(num_samples/10));  % 计算进度显示的步长
    
    for i = 1:num_samples
        % 显示进度
        if mod(i, progress_step) == 0
            fprintf('已完成: %d/%d (%.1f%%)\n', i, num_samples, 100*i/num_samples);
        end
        
        % 随机选择输入输出符号数，但保持在合理范围内
        num_x = randi([2, min(8, max_input_size)]);  % 随机选择输入符号数
        num_y = randi([2, min(8, max_output_size)]);  % 随机选择输出符号数
        
        % 生成随机信道转移矩阵
        W = rand(num_x, num_y);  % 生成随机矩阵
        % 确保每行和为1
        W = W ./ sum(W, 2);  % 对每行归一化
        
        % 使用传统BA算法找到最优输入分布和信道容量
        [C, optimal_p, iter] = arimoto_blahut(W, 1e-6, 3000);  % 运行BA算法
        
        % 如果未收敛，尝试重新生成更简单的矩阵
        if iter >= 3000  % 如果达到最大迭代次数
            % 使用更平均的分布生成新矩阵
            W = rand(num_x, num_y) + 0.5;  % 添加基线值使分布更均匀
            W = W ./ sum(W, 2);  % 对每行归一化
            [C, optimal_p, ~] = arimoto_blahut(W, 1e-6, 3000);  % 重新运行BA算法
        end
        
        % 保存数据
        training_data{i}.W = W;  % 保存信道矩阵
        training_data{i}.optimal_p = optimal_p;  % 保存最优输入分布
        training_data{i}.capacity = C;  % 保存信道容量
    end
    
    fprintf('训练数据生成完成!\n');  % 输出完成消息
end
