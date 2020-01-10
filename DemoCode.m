clear;
syms x;
% 产生模拟的数据
N = 30;
JitterVale = 0.05;
load('DemoData.mat');
rng(randseeds);
% 若无法下载样例数据，请运行下面三行打码，产生新的随机数据。
% 但，需要注意的是，由于新数据具有很强随机性且8个正弦函数不会适用于所有随机数据，
% 故在无样例数据的情况下，需要多次尝试，才能得到文章中的数据结果。
% x =  linspace(0,1,N)';
% y0 =  (x-0.5).^2;
% y = y0 + (2*(rand(length(x),1)-0.5))*JitterVale;
xNew = linspace(0,1,N*10)';

%% 定义一个比真实模型更加复杂的模型
StartPoint = [1,...%a
    1.627, 2.757, -0.07503,...% a1
    1.62, 3.034, 2.945,...% b1
    0.02967, 14.44, 0.2716,...% c1
    0.02446, 17.33, 1.764,...% d1
    0.0168, 22.73, 1.729,...% e1
    0.0192, 63.32, 0.03989,...% f1
    0.01574, 67.17, 1.738,...% g1
    0.01547, 90.65, 0.5087,...% h1
    ];

s = fitoptions('Method','NonlinearLeastSquares',...
    'Startpoint',StartPoint);
ft = fittype(['a*((x-0.5)^2)+',...
    'a1*sin(a2*x+a3)+ b1*sin(b2*x+b3)+c1*sin(c2*x+c3)+d1*sin(d2*x+d3)+',...
    'e1*sin(e2*x+e3)+ f1*sin(f2*x+f3)+g1*sin(g2*x+g3)+h1*sin(h2*x+h3)'],...
    'independent','x','option',s);

%% 传统模型拟合的方法
[cfun,gof]=fit(x,y,ft);
yi = cfun(xNew);

%% 机器学习的拟合方法
NLeave = 10;
cfunML = cell(NLeave,1);
gofML = cell(NLeave,1);
MSEML_Preict = nan(NLeave,1);
for i = 1:NLeave
    % 采用交叉验证进行拟合
    [trainID,testID] = crossvalind('LeaveMOut',N,4);
    trainDataX = x(trainID);
    trainDataY = y(trainID);
    [cfunML{i},gofML{i}]=fit(trainDataX,trainDataY,ft);
    % 使用训练集对训练的模型进行检验，计算预测误差
    testDataX = x(testID);
    testDataY = y(testID);
    PredicDataY = cfunML{i}(testDataX);
    MSEML_Preict(i) = sum((PredicDataY-testDataY).^2)/length(testDataX);
end
% 依据最小误差，选择最佳的拟合模型
OptimalIndex = MSEML_Preict == min(MSEML_Preict);
yii = cfunML{OptimalIndex}(xNew);
% 增加样本量
x2 =  linspace(0,1,N*4)';
y20 =  (x2-0.5).^2;
y2 = y20 + (2*(rand(length(x2),1)-0.5))*JitterVale;
[cfun2,gof2]=fit(x2,y2,ft);
y2i = cfun2(xNew);


%% 显示结果
% 输出数据
clc;
disp('====传统拟合方法结果=====');
disp(cfun);disp(gof);

disp('====交叉验证结果=====');
disp(cfunML{OptimalIndex});disp(gofML{OptimalIndex});

disp('====增大样本量结果=====');
disp(cfun2);disp(gof2);

% 画图
figure;hold on;
subplot(2,2,1);axis square;hold on;
plot(x,y,'r*',x,y0,'k');ylabel('y');
title(['(a) 真实模型：y = (x-0.5)^2']);
ylim([-1*JitterVale,mean(x)^2 + JitterVale]);
set(gca,'xtick',[]);

subplot(2,2,2);axis square;hold on;
plot(x,y,'r*',xNew,yi,'k');
title(['(b) 传统方法']);
ylim([-1*JitterVale, (mean(x)^2 + JitterVale)]);
set(gca,'xtick',[],'ytick',[]);

subplot(2,2,3);axis square;hold on;
plot(x,y,'r*',xNew,yii,'k');
xlabel('x');ylabel('y');
title(['(c) 留四交叉验证']);
ylim([-1*JitterVale,mean(x)^2 + JitterVale]);

subplot(2,2,4);axis square;hold on;
plot(x2,y2,'r*',xNew,y2i,'k');
xlabel('x');
title(['(d) 样本量增加3倍']);
ylim([-1*JitterVale,mean(x)^2 + JitterVale]);
set(gca,'ytick',[]);

% 注：本程序测试的matlab版本为'9.5.0.944444 (R2018b)'.

