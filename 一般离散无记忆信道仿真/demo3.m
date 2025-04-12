clear; clc;

% 定义信道条件概率矩阵 P(Y|X)
P_Y_given_X = [
  0.3, 0.1, 0.4, 0.2;
  0.2, 0.5, 0.1, 0.2;
  0.1, 0.3, 0.3, 0.3;
  0.4, 0.2, 0.1, 0.3
];


%% 方法1：Arimoto-Blahut 算法
fprintf('==================== 方法1：Arimoto-Blahut 算法 =====================\n');

% 调用 Arimoto-Blahut 算法
[C_arimoto, optimal_p_arimoto, iterations_arimoto] = arimoto_blahut(P_Y_given_X);

% 显示结果
fprintf('信道容量 C: %.4f 比特/符号\n', C_arimoto);
fprintf('最佳输入分布 p: [');
fprintf('%.4f ', optimal_p_arimoto);
fprintf(']\n');
fprintf('迭代次数: %d 次\n', iterations_arimoto);

fprintf('-----------------------------------------------------------------------\n\n');

%% 方法2：Acc_Arimoto-Blahut 算法
fprintf('==================== 方法2:Acc_Arimoto-Blahut 算法 =====================\n');

% 调用 Arimoto-Blahut 算法
[C_arimoto, optimal_p_arimoto, iterations_arimoto] = accelerated_arimoto_blahut(P_Y_given_X);

% 显示结果
fprintf('信道容量 C: %.4f 比特/符号\n', C_arimoto);
fprintf('最佳输入分布 p: [');
fprintf('%.4f ', optimal_p_arimoto);
fprintf(']\n');
fprintf('迭代次数: %d 次\n', iterations_arimoto);

fprintf('-----------------------------------------------------------------------\n\n');
%% 方法3：arimoto_blahut_newton 算法
fprintf('==================== 方法3:arimoto_blahut_newton 算法 =====================\n');

% 调用 natural_gradient_ba 算法
[C_arimoto, optimal_p_arimoto, iterations_arimoto] = arimoto_blahut_newton(P_Y_given_X);

% 显示结果
fprintf('信道容量 C: %.4f 比特/符号\n', C_arimoto);
fprintf('最佳输入分布 p: [');
fprintf('%.4f ', optimal_p_arimoto);
fprintf(']\n');
fprintf('迭代次数: %d 次\n', iterations_arimoto);

fprintf('-----------------------------------------------------------------------\n\n');
%% 方法4：arimoto_blahut_adam 算法
fprintf('==================== 方法4:arimoto_blahut_adam 算法 =====================\n');

% 调用 natural_gradient_ba 算法
[C_arimoto, optimal_p_arimoto, iterations_arimoto] = arimoto_blahut_adam(P_Y_given_X);

% 显示结果
fprintf('信道容量 C: %.4f 比特/符号\n', C_arimoto);
fprintf('最佳输入分布 p: [');
fprintf('%.4f ', optimal_p_arimoto);
fprintf(']\n');
fprintf('迭代次数: %d 次\n', iterations_arimoto);

fprintf('-----------------------------------------------------------------------\n\n');
%% 方法5：神经网络算法
fprintf('==================== 方法5:arimoto_blahut_with_nn 算法 =====================\n');

% 调用 natural_gradient_ba 算法
[C_arimoto, optimal_p_arimoto, iterations_arimoto] = arimoto_blahut_with_nn(P_Y_given_X);

% 显示结果
fprintf('信道容量 C: %.4f 比特/符号\n', C_arimoto);
fprintf('最佳输入分布 p: [');
fprintf('%.4f ', optimal_p_arimoto);
fprintf(']\n');
fprintf('迭代次数: %d 次\n', iterations_arimoto);

fprintf('-----------------------------------------------------------------------\n\n');