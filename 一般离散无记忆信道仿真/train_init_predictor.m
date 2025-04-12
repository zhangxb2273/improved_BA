% 3. 训练神经网络的函数
function trained_model = train_ba_initialization_model(training_data)
% TRAIN_BA_INITIALIZATION_MODEL 训练神经网络以预测BA算法的初始输入分布
%
% 输入参数:
%   training_data - 包含W矩阵和最优p分布的训练数据
%
% 输出参数:
%   trained_model - 训练好的神经网络模型

    % 创建网络
    net = create_ba_initialization_network();
    
    % 准备训练数据
    numObservations = length(training_data);
    X = cell(numObservations, 1);
    Y = cell(numObservations, 1);
    
    for i = 1:numObservations
        W = training_data{i}.W;
        optimal_p = training_data{i}.optimal_p;
        
        % 处理输入数据
        W_flat = W(:)';
        if length(W_flat) < net.Layers(1).InputSize
            W_flat = [W_flat, zeros(1, net.Layers(1).InputSize - length(W_flat))];
        else
            W_flat = W_flat(1:net.Layers(1).InputSize);
        end
        
        X{i} = W_flat;
        
        % 处理目标数据
        if length(optimal_p) < net.Layers(end-1).OutputSize
            Y{i} = [optimal_p; zeros(net.Layers(end-1).OutputSize - length(optimal_p), 1)];
        else
            Y{i} = optimal_p(1:net.Layers(end-1).OutputSize);
        end
    end
    
    % 设置训练选项
    options = trainingOptions('adam', ...
        'MaxEpochs', 100, ...
        'MiniBatchSize', 32, ...
        'InitialLearnRate', 0.001, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.1, ...
        'LearnRateDropPeriod', 20, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', true, ...
        'Plots', 'training-progress', ...
        'ValidationFrequency', 30);
    
    % 训练网络
    trained_model = trainNetwork(X, Y, net.Layers, options);
end


function net = create_ba_initialization_network()
    % 创建一个前馈神经网络用于预测BA算法的初始输入分布
    
    % 假设最大输入符号数为64（可以根据实际需求调整）
    max_input_size = 64;
    max_output_size = 64;
    
    % 输入层大小为转移矩阵展平后的大小
    input_size = max_input_size * max_output_size;
    
    % 创建神经网络架构
    layers = [
        featureInputLayer(input_size, 'Name', 'input')
        fullyConnectedLayer(256, 'Name', 'fc1')
        batchNormalizationLayer('Name', 'bn1')
        reluLayer('Name', 'relu1')
        dropoutLayer(0.3, 'Name', 'drop1')
        fullyConnectedLayer(128, 'Name', 'fc2')
        batchNormalizationLayer('Name', 'bn2')
        reluLayer('Name', 'relu2')
        fullyConnectedLayer(64, 'Name', 'fc3')
        batchNormalizationLayer('Name', 'bn3')
        reluLayer('Name', 'relu3')
        fullyConnectedLayer(max_input_size, 'Name', 'fc_out')
        softmaxLayer('Name', 'softmax')
    ];
    
    net = dlnetwork(layers);
end


function training_data = generate_training_data(num_samples, max_input_size, max_output_size)
% GENERATE_TRAINING_DATA 生成用于训练神经网络的数据集
%
% 输入参数:
%   num_samples    - 要生成的样本数量
%   max_input_size - 最大输入符号数
%   max_output_size- 最大输出符号数
%
% 输出参数:
%   training_data  - 包含W矩阵和最优p分布的训练数据

    training_data = cell(num_samples, 1);
    
    for i = 1:num_samples
        % 随机选择输入输出符号数
        num_x = randi([2, max_input_size]);
        num_y = randi([2, max_output_size]);
        
        % 生成随机信道转移矩阵
        W = rand(num_x, num_y);
        % 确保每行和为1
        W = W ./ sum(W, 2);
        
        % 使用传统BA算法找到最优输入分布和信道容量
        [C, optimal_p, ~] = arimoto_blahut(W, 1e-8, 1000);
        
        % 保存数据
        training_data{i}.W = W;
        training_data{i}.optimal_p = optimal_p;
        training_data{i}.capacity = C;
    end
end
