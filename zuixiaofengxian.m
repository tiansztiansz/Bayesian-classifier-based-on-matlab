clear;
clc;

% 总训练样本数
N = 29;

% 类别数目
w = 4;

% 每一个样本的特征数
n = 3;

% 训练样本中每一类的数目
N1 = 4; N2 = 7; N3 = 8; N4 = 10;

%% 训练数据（注意这里列为记录，行为特征）
% A 属于类别 w1。这里就表示 A 有 3 个特征，4条记录
A = [
    864.45 877.88 1418.79 1449.58;
    1647.31 2031.66 1775.89 1641.58;
    2665.9 3071.18 2772.9 3045.12
    ];
% B 属于类别 w2。这里就表示 B 有3个特征，7条记录
B = [
    2352.12 2297.28 2092.62 2205.36 2949.16 2802.88 2063.54;
    2557.04 3340.14 3177.21 3243.74 3244.44 3017.11 3199.76;
    1411.53 535.62 584.32 1202.69 662.42 1984.98 1257.21
    ];
% C 属于类别 w3。这里就表示C有3个特征，8条记录
C = [
    1739.94 1756.77 1803.58 1571.17 1845.59 1692.62 1680.67 1651.52;
    1675.15 1652 1583.12 1731.04 1918.81 1867.5 1575.78 1713.28;
    2395.96 1514.98 2163.05 1735.33 2226.49 2108.97 1725.1 1570.38
    ];
% D 属于类别 w4。这里就表示D有3个特征，10条记录
D = [
    373.3 222.85 401.3 363.34 104.8 499.85 172.78 341.59 291.02 237.63;
    3087.05 3059.54 3259.94 3477.95 3389.83 3305.75 3084.49 3076.62 3095.68 3077.78;
    2429.47 2002.33 2150.98 2462.86 2421.83 3196.22 2328.65 2438.63 2088.95 2251.96
    ];

% 计算每一类的平均值
X1 = mean(A')'
X2 = mean(B')'
X3 = mean(C')'
X4 = mean(D')'

% 每种类型的训练样本的协方差矩阵
S1 = cov(A')
S2 = cov(B')
S3 = cov(C')
S4 = cov(D')

% 每种类型的训练样本的逆矩阵
S1_ = inv(S1)
S2_ = inv(S2)
S3_ = inv(S3)
S4_ = inv(S4)

% 方差矩阵的行列式
S11 = det(S1)
S22 = det(S2)
S33 = det(S3)
S44 = det(S4)

% 先验概率
Pw1 = N1 / N
Pw2 = N2 / N
Pw3 = N3 / N
Pw4 = N4 / N

% 测试数据。注意这里的数据和上面的训练数据样式有一点差别，这里行是记录、列是特征
sample = [1702.8 1639.79 2068.74
    1877.93 1860.96 1975.3
    867.81 2334.68 2535.1
    1831.49 1713.11 1604.68
    460.69 3274.77 2172.99
    2374.98 3346.98 975.31
    2271.89 3482.97 946.7
    1783.64 1597.99 2261.31
    198.83 3250.45 2445.08
    1494.63 2072.59 2550.51
    1597.03 1921.52 2126.76
    1598.93 1921.08 1623.33
    1243.13 1814.07 3441.07
    2336.31 2640.26 1599.63
    354 3300.12 2373.61
    2144.47 2501.62 591.51
    426.31 3105.29 2057.8
    1507.13 1556.89 1954.51
    343.07 3271.72 2036.94
    2201.94 3196.22 935.53
    2232.43 3077.87 1298.87
    1580.1 1752.07 2463.04
    1962.4 1594.97 1835.95
    1495.18 1957.44 3498.02
    1125.17 1594.39 2937.73
    24.22 3447.31 2145.01
    1269.07 1910.72 2701.97
    1802.07 1725.81 1966.35
    1817.36 1927.4 2328.79
    1860.45 1782.88 1875.13];

% 定义风险损失函数（这里的损失函数为对角线元素全为 0，其它元素全为 1 的矩阵）
loss = ones(4) - diag(diag(ones(4)));
figure(1)
plot(loss);
grid on;
title('第1个损失函数');
xlabel('类别'); ylabel('损失函数的值');

% 定义另一种风险损失函数（这里的损失函数为对角线元素全为 0，对角线上方元素全为 1，对角线下方元素全为 2）
lossTwo = (ones(4) - diag(diag(ones(4)))) + tril(ones(4), -1);
figure(2)
plot(lossTwo);
grid on;
title('第2个损失函数');
xlabel('类别'); ylabel('损失函数的值');

% 对每一个测试样本进行循环
for k = 1:30
    % 计算后验概率
    P1 = -1/2 * (sample(k, :)' - X1)' * S1_ * (sample(k, :)' - X1) + log(Pw1) - 1/2 * log(S11);
    P2 = -1/2 * (sample(k, :)' - X2)' * S2_ * (sample(k, :)' - X2) + log(Pw2) - 1/2 * log(S22);
    P3 = -1/2 * (sample(k, :)' - X3)' * S3_ * (sample(k, :)' - X3) + log(Pw3) - 1/2 * log(S33);
    P4 = -1/2 * (sample(k, :)' - X4)' * S4_ * (sample(k, :)' - X4) + log(Pw4) - 1/2 * log(S44);

    % 基于第 1 个风险函数计算每 1 个样本在每 1 类上的风险
    risk1 = loss(1, 1) * P1 + loss(1, 2) * P2 + loss(1, 3) * P3 + loss(1, 4) * P4;
    risk2 = loss(2, 1) * P1 + loss(2, 2) * P2 + loss(2, 3) * P3 + loss(2, 4) * P4;
    risk3 = loss(3, 1) * P1 + loss(3, 2) * P2 + loss(3, 3) * P3 + loss(3, 4) * P4;
    risk4 = loss(4, 1) * P1 + loss(4, 2) * P2 + loss(4, 3) * P3 + loss(4, 4) * P4;
    % 寻找每 1 个样本的最小风险
    risk = [risk1 risk2 risk3 risk4]
    minriskloss = min(risk)

    % 基于第 2 个风险函数计算每 1 个样本在每 1 类上的风险
    riskTwo1 = lossTwo(1, 1) * P1 + lossTwo(1, 2) * P2 + lossTwo(1, 3) * P3 + lossTwo(1, 4) * P4;
    riskTwo2 = lossTwo(2, 1) * P1 + lossTwo(2, 2) * P2 + lossTwo(2, 3) * P3 + lossTwo(2, 4) * P4;
    riskTwo3 = lossTwo(3, 1) * P1 + lossTwo(3, 2) * P2 + lossTwo(3, 3) * P3 + lossTwo(3, 4) * P4;
    riskTwo4 = lossTwo(4, 1) * P1 + lossTwo(4, 2) * P2 + lossTwo(4, 3) * P3 + lossTwo(4, 4) * P4;
    % 寻找每 1 个样本的最小风险
    riskTwo = [riskTwo1 riskTwo2 riskTwo3 riskTwo4]
    minrisklossTwo = min(riskTwo)

    % 对于第 1 个风险函数，判断每 1 个样本在哪一类中风险最小，就将它归为哪一类
    if risk1 == min(risk)
        w = 1
        figure(3)
        plot3(sample(k, 1), sample(k, 2), sample(k, 3), 'ro'); grid on; hold on;
        title('基于第1个风险函数的最小风险的分类结果示意图');
        xlabel('第一特征坐标');
        ylabel('第二特征坐标');
        zlabel('第三特征坐标');
    elseif risk2 == min(risk)
        w = 2
        figure(3)
        plot3(sample(k, 1), sample(k, 2), sample(k, 3), 'b>'); grid on; hold on;
        title('基于第1个风险函数的最小风险的分类结果示意图');
        xlabel('第一特征坐标');
        ylabel('第二特征坐标');
        zlabel('第三特征坐标');
    elseif risk3 == min(risk)
        w = 3
        figure(3)
        plot3(sample(k, 1), sample(k, 2), sample(k, 3), 'g+'); grid on; hold on;
        title('基于第1个风险函数的最小风险的分类结果示意图');
        xlabel('第一特征坐标');
        ylabel('第二特征坐标');
        zlabel('第三特征坐标');
    elseif risk4 == min(risk)
        w = 4
        figure(3)
        plot3(sample(k, 1), sample(k, 2), sample(k, 3), 'y*'); grid on; hold on;
        title('基于第1个风险函数的最小风险的分类结果示意图');
        xlabel('第一特征坐标');
        ylabel('第二特征坐标');
        zlabel('第三特征坐标');
    else
        return
    end

    % 对于第 2 个风险函数，判断每 1 个样本在哪一类中风险最小，就将它归为哪一类
    if riskTwo1 == min(riskTwo)
        w = 1
        figure(4)
        plot3(sample(k, 1), sample(k, 2), sample(k, 3), 'ro'); grid on; hold on;
        title('基于第2个风险函数的最小风险的分类结果示意图');
        xlabel('第一特征坐标');
        ylabel('第二特征坐标');
        zlabel('第三特征坐标');
    elseif riskTwo2 == min(riskTwo)
        w = 2
        figure(4)
        plot3(sample(k, 1), sample(k, 2), sample(k, 3), 'b>'); grid on; hold on;
        title('基于第2个风险函数的最小风险的分类结果示意图');
        xlabel('第一特征坐标');
        ylabel('第二特征坐标');
        zlabel('第三特征坐标');
    elseif riskTwo3 == min(riskTwo)
        w = 3
        figure(4)
        plot3(sample(k, 1), sample(k, 2), sample(k, 3), 'g+'); grid on; hold on;
        title('基于第2个风险函数的最小风险的分类结果示意图');
        xlabel('第一特征坐标');
        ylabel('第二特征坐标');
        zlabel('第三特征坐标');
    elseif riskTwo4 == min(riskTwo)
        w = 4
        figure(4)
        plot3(sample(k, 1), sample(k, 2), sample(k, 3), 'y*'); grid on; hold on;
        title('基于第2个风险函数的最小风险的分类结果示意图');
        xlabel('第一特征坐标');
        ylabel('第二特征坐标');
        zlabel('第三特征坐标');
    else
        return
    end

end
