%%
clc, clear, close all
%% kreiranje funkcije koja treba da se fituje
N = 1500;
x = linspace(0,0.5,N);
h = sin(40*pi * x) + 3*sin(18*pi * x);
std = 0.2;
y = h + std*randn(1,N) ; % f_sum
%% prikaz
figure, hold all
plot(x, y,'b','LineWidth', 1);
plot(x, h,'r','LineWidth', 3);
legend('y(x)','h(x)');
%% ulaz = x, izlaz = y
ulaz = x;
izlaz = y;
%% Kreiranje neuralne mreze
net = fitnet([10 6]);
net.divideFcn = ''; % iskljucena zastita od preobucavanja
net.trainFcn = 'trainlm';

net.trainParam.epochs = 3000;
net.trainParam.goal = 1e-3;
net.trainParam.min_grad = 1e-4; 

%% Treniranje neuralne mreze
net = train(net, ulaz, izlaz);
%% Performanse neuralne mreze
pred = sim(net, ulaz);
%% prikaz
figure, hold all
plot(ulaz, izlaz,'b','LineWidth', 1)
plot(ulaz, pred, 'r','LineWidth',3);
%plot(ulaz, h, 'r','LineWidth', 3)
legend('y(x)','pred');