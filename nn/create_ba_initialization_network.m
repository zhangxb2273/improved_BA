function layers = create_ba_initialization_network(input_size, output_size)
% CREATE_BA_INITIALIZATION_NETWORK 创建用于预测BA算法初始分布的神经网络
%
% 该函数构建一个前馈神经网络，用于预测Blahut-Arimoto算法的优化初始输入分布
%
% 输入参数:
%   input_size  - 输入层大小（展平的信道矩阵大小）
%   output_size - 输出层大小（最大输入符号数）
%
% 输出参数:
%   layers - MATLAB深度学习网络层数组
    
    % 创建简单的神经网络架构
    layers = [
        featureInputLayer(input_size)  % 输入层，接受展平的信道矩阵
        fullyConnectedLayer(128)       % 第一个全连接层，128个神经元
        reluLayer                      % ReLU激活函数
        fullyConnectedLayer(64)        % 第二个全连接层，64个神经元
        reluLayer                      % ReLU激活函数
        fullyConnectedLayer(output_size)  % 输出层，大小为最大输入符号数
        regressionLayer                % 回归输出层，用于预测连续值
    ];
end
