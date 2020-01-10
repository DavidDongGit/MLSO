clear;
syms x;
% ����ģ�������
N = 30;
JitterVale = 0.05;
load('DemoData.mat');
rng(randseeds);
% ���޷������������ݣ��������������д��룬�����µ�������ݡ�
% ������Ҫע����ǣ����������ݾ��к�ǿ�������8�����Һ�����������������������ݣ�
% �������������ݵ�����£���Ҫ��γ��ԣ����ܵõ������е����ݽ����
% x =  linspace(0,1,N)';
% y0 =  (x-0.5).^2;
% y = y0 + (2*(rand(length(x),1)-0.5))*JitterVale;
xNew = linspace(0,1,N*10)';

%% ����һ������ʵģ�͸��Ӹ��ӵ�ģ��
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

%% ��ͳģ����ϵķ���
[cfun,gof]=fit(x,y,ft);
yi = cfun(xNew);

%% ����ѧϰ����Ϸ���
NLeave = 10;
cfunML = cell(NLeave,1);
gofML = cell(NLeave,1);
MSEML_Preict = nan(NLeave,1);
for i = 1:NLeave
    % ���ý�����֤�������
    [trainID,testID] = crossvalind('LeaveMOut',N,4);
    trainDataX = x(trainID);
    trainDataY = y(trainID);
    [cfunML{i},gofML{i}]=fit(trainDataX,trainDataY,ft);
    % ʹ��ѵ������ѵ����ģ�ͽ��м��飬����Ԥ�����
    testDataX = x(testID);
    testDataY = y(testID);
    PredicDataY = cfunML{i}(testDataX);
    MSEML_Preict(i) = sum((PredicDataY-testDataY).^2)/length(testDataX);
end
% ������С��ѡ����ѵ����ģ��
OptimalIndex = MSEML_Preict == min(MSEML_Preict);
yii = cfunML{OptimalIndex}(xNew);
% ����������
x2 =  linspace(0,1,N*4)';
y20 =  (x2-0.5).^2;
y2 = y20 + (2*(rand(length(x2),1)-0.5))*JitterVale;
[cfun2,gof2]=fit(x2,y2,ft);
y2i = cfun2(xNew);


%% ��ʾ���
% �������
clc;
disp('====��ͳ��Ϸ������=====');
disp(cfun);disp(gof);

disp('====������֤���=====');
disp(cfunML{OptimalIndex});disp(gofML{OptimalIndex});

disp('====�������������=====');
disp(cfun2);disp(gof2);

% ��ͼ
figure;hold on;
subplot(2,2,1);axis square;hold on;
plot(x,y,'r*',x,y0,'k');ylabel('y');
title(['(a) ��ʵģ�ͣ�y = (x-0.5)^2']);
ylim([-1*JitterVale,mean(x)^2 + JitterVale]);
set(gca,'xtick',[]);

subplot(2,2,2);axis square;hold on;
plot(x,y,'r*',xNew,yi,'k');
title(['(b) ��ͳ����']);
ylim([-1*JitterVale, (mean(x)^2 + JitterVale)]);
set(gca,'xtick',[],'ytick',[]);

subplot(2,2,3);axis square;hold on;
plot(x,y,'r*',xNew,yii,'k');
xlabel('x');ylabel('y');
title(['(c) ���Ľ�����֤']);
ylim([-1*JitterVale,mean(x)^2 + JitterVale]);

subplot(2,2,4);axis square;hold on;
plot(x2,y2,'r*',xNew,y2i,'k');
xlabel('x');
title(['(d) ����������3��']);
ylim([-1*JitterVale,mean(x)^2 + JitterVale]);
set(gca,'ytick',[]);

% ע����������Ե�matlab�汾Ϊ'9.5.0.944444 (R2018b)'.

