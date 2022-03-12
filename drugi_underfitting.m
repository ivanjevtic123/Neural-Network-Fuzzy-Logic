%%
clc, clear, close all
%%
load('dataset2.mat');
%% transponovanje matrice i iscrtavanje klasa
ob = pod(:, 1:2)';
klasa = pod(:, 3)';
N = length(klasa);

K1 = ob(:, klasa == 1);
K2 = ob(:, klasa == 2);
K3 = ob(:, klasa == 3);

figure, hold all
plot(K1(1, :), K1(2, :), 'o')
plot(K2(1, :), K2(2, :), '*')
plot(K3(1, :), K3(2, :), 's')
%% One-hot encoding
izlaz = zeros(3, N);

izlaz(1, klasa == 1) = 1;
izlaz(2, klasa == 2) = 1;
izlaz(3, klasa == 3) = 1;

ulaz = ob;
%% podela podataka na trening i test skup
ind = randperm(N);
indTrening = ind(1 : 0.9*N);
indTest = ind(0.9*N+1 : N);

ulazTrening = ulaz(:, indTrening);
izlazTrening = izlaz(:, indTrening);

ulazTest = ulaz(:, indTest);
izlazTest = izlaz(:, indTest);
%% Kreiranje NM
arhitektura3 = [3];
net = patternnet(arhitektura3);

for i = 1 : length(arhitektura3)
    net.layers{i}.transferFcn = 'tansig';
end

net.layers{length(arhitektura3) + 1}.transferFcn = 'softmax';
net.performFcn = 'crossentropy';
net.divideFcn = '';

net.trainParam.epochs = 1000;
net.trainParam.goal = 1e-4;
net.trainParam.min_grad = 1e-5;
%% Treniranje NM
net = train(net, ulazTrening, izlazTrening);
%% Performanse NM
predTest = net(ulazTest);
predTrening = net(ulazTrening);
%%
figure, plotconfusion(izlazTest, predTest)
[c,cm] = confusion(izlazTest,predTest);
cm = cm';
Rtest = cm(1,1)/(cm(1,1)+cm(2,1)+cm(3,1));
Ptest = cm(1,1)/(cm(1,1)+cm(1,2)+cm(1,3));
%%
figure, plotconfusion(izlazTrening, predTrening)
[c,cm] = confusion(izlazTrening,predTrening);
cm = cm';
Rtrening = cm(1,1)/(cm(1,1)+cm(2,1)+cm(3,1));
Ptrening = cm(1,1)/(cm(1,1)+cm(1,2)+cm(1,3));
%% Granica odlucivanja
Ntest = 500;
x1 = repmat(linspace(-4, 4, Ntest), 1, Ntest);
x2 = repelem(linspace(-4, 4, Ntest), Ntest);
ulazGO = [x1; x2];

predGO = net(ulazGO);
[~, klasaGO] = max(predGO);

%%
K1go = ulazGO(:, predGO(1, :) >= 0.7);
K2go = ulazGO(:, predGO(2, :) >= 0.7);
K3go = ulazGO(:, predGO(3, :) >= 0.7);

figure, hold all
plot(K1go(1, :), K1go(2, :), '.')
plot(K2go(1, :), K2go(2, :), '.')
plot(K3go(1, :), K3go(2, :), '.')
plot(K1(1, :), K1(2, :), 'bo')
plot(K2(1, :), K2(2, :), 'r*')
plot(K3(1, :), K3(2, :), 'ys')
legend('K1','K2','K3');