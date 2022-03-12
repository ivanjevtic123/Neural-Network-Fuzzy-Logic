%%
clc, clear, close all
%% odvajanje klasa
trening = Stardataset';
izlaz = trening(9,:);
K1 = trening(1:8,izlaz == 1);
K2 = trening(1:8,izlaz == 0);
%% Prikaz podataka
%figure,hold all
%plot(K1(1,:),K1(2,:),'o');
%plot(K2(1,:),K2(2,:),'*');
%legend('K1 = 1','K2 = 0');
figure,hold all
bar([length(K1),length(K2)]);
%legend('K1 = 1';'K2 = 0');
%% podela na trening,validacioni i test skup
N1 = length(K1);
N2 = length(K2);
%% podela na trening,validacioni i test skup
K1trening = K1(:,1:0.8*N1);
K1val = K1(:,0.8*N1 + 1 : 0.9*N1+1);
K1test = K1(:,0.9*N1+1:N1 + 1);

K2trening = K2(:,1:0.8*N2 + 1);
K2val = K2(:,0.8*N2 + 2: 0.9*N2 + 1);
K2test = K2(:,0.9*N2 + 1:N2);
%%
ulazTrening = [K1trening,K2trening];
izlazTrening = [ones(1,length(K1trening)),zeros(1,length(K2trening))]; % mora celi broj kao argument pa ne moze sa mnozenjem 0.x

ulazVal = [K1val,K2val];
izlazVal = [ones(1,length(K1val)),zeros(1,length(K2val))];

%% 
indTest = randperm(length(izlazTrening));
ulazTrening = ulazTrening(:,indTest);
izlazTrening = izlazTrening(indTest);

indVal = randperm(length(izlazVal));
ulazVal = ulazVal(:,indVal);
izlazVal = izlazVal(indVal);

ulazSve = [ulazTrening,ulazVal]; %90 posto za trening i validaciju,ostalo za test
izlazSve = [izlazTrening,izlazVal];
N = length(izlazSve);
%%
best_F1 = 0 ; 
best_struct = [10,10];
best_trainFcn = 'tansig';
best_reg = 0.2;
best_w = 2;
%%
for structure = {[2],[10,10],[20,15],[3,3,5]}
    for trainFcn = {'logsig','tansig'}
        for reg = {0.8,0.4,0.2}
            for weight = {2,4,6}
                net = feedforwardnet(structure{1}, 'trainlm');
           
                for i = 1 : length(structure)
                net.layers{i}.transferFcn = trainFcn{1};
                end
               
                net.performParam.regularization = reg{1};
                net.trainFcn = 'trainscg';
                
                net.trainParam.epochs = 1000;
                net.divideFcn = 'divideint';
                
                net.divideParam.trainInd = 1:length(ulazTrening); %80 posto za trening
                net.divideParam.valInd =  length(ulazTrening) + 1: length(ulazTrening) + length(ulazVal);%10 posto za validaciju
                net.divideParam.testInd = [];             
                
               % net.trainParam.showWindow = false;
               % net.trainParam.showCommandLine = true;
                w = ones(1,N);
                w(izlazSve == 1) = weight{1};
                
               net.trainParam.epochs = 1000;
               net.trainParam.goal = 1e-4;
               net.trainParam.min_grad = 1e-5;
               net.trainParam.max_fail = 10;
                
               [net,tr] = train(net,ulazSve,izlazSve,[],[],w);

               predVal = net(ulazVal);
               [c,cm] = confusion(izlazVal,predVal);
               cm = cm';
               R = cm(2,2)/(cm(2,2)+cm(1,2));
               P = cm(2,2)/(cm(2,2)+cm(2,1));
               F1 = 2*P*R/(P+R);
                if best_F1 < F1
                    best_F1 = F1 ; 
                    best_struct = structure{1};
                    best_trainFcn = trainFcn{1};
                    best_reg = reg{1};
                    best_w = weight{1};
                end
            end
        end
    end
end

%%
 disp("");
 disp("Rezultat treniranja: ");
 disp("Najbolja struktura mreze:  " + num2str(best_struct));
 disp("Najbolja aktivaciona funkcija:  " + string(best_trainFcn));
 disp("Najbolji koeficijent regularizacije:  " + string(best_reg));
 disp("Najbolja tezina:  " + string(best_w));
 disp("Najbolji f1:  " + string(best_F1));

%% Ponovno treniranje NM 
net = feedforwardnet(best_struct, 'trainlm');


 for i = 1 : length(structure)
  net.layers{i}.transferFcn = best_trainFcn;
 end
    
net.performParam.regularization = best_reg;
net.trainFcn = 'trainscg';
w = ones(1,length(ulazSve));
w(izlazSve==1) = best_w;

net.trainParam.epochs = 1000;
net.trainParam.goal = 1e-4;
net.trainParam.min_grad = 1e-5;
net.trainParam.max_fail = 10;

[net,tr] = train (net,ulazSve,izlazSve,[],[],w); % u tr se cuvaju razni parametri,kao max br epoha do preobucavanja

%net.trainParam.showWindow = false;
%net.trainParam.showCommandLine = true;

%% Provera performansi NM za trening skup 
 predSve = net(ulazSve);
 [c,cm] = confusion(izlazSve,predSve);
 cm = cm';
 R = cm(2,2)/(cm(2,2)+cm(1,2));
 P = cm(2,2)/(cm(2,2)+cm(2,1));
 F1 = 2*P*R/(P+R);
 figure,plotconfusion(izlazSve,predSve);
%% Formiranje test skupa
ulazTest = [K1test,K2test];
izlazTest = [ones(1,length(K1test)),zeros(1,length(K2test))];

indTest = randperm(length(izlazTest));
ulazTest = ulazTest(:,indTest);
izlazTest = izlazTest(indTest);
%% Provera performansi NM za test skup
predTest = net(ulazTest);
[c,cm] = confusion(izlazTest,predTest);
cm = cm';
 R = cm(2,2)/(cm(2,2)+cm(1,2));
 P = cm(2,2)/(cm(2,2)+cm(2,1));
 F1 = 2*P*R/(P+R);
figure,plotconfusion(izlazTest,predTest);
%%
%figure,hold all
%plot(K1test(1,:),K1test(2,:),'o');
%plot(K2test(1,:),K2test(2,:),'*');
%plot(K1(1,:),K1(2,:),'.');
%plot(K2(1,:),K2(2,:),'.');

%legend('K1test','K2test','K1','K2');



