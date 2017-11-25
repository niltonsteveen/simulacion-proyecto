clc
clear all
% close all
load('sensor_data.csv');
%Se cargan las muestras
X=sensor_data(1:end,1:end-1);
%Se cargan las salidas
Y=sensor_data(1:end,end);
Y2=(Y')';
Rept=10;
EficienciaTest=zeros(1,Rept);
PrecisionTest=zeros(Rept,4);
SensibilidadTest=zeros(Rept,4);
boxConstraint=100; % Este es el parametro de regularización
gamma=100; % Este es el parametro (si lo necesita) de la funcion kernel

N=size(X,1); %Numero de muestras
NClases=length(unique(Y));

Tipo=input('Ingrese según lo desee: \n 1. Funciones Discriminantes Gausianas\n 2. K-nn\n 3. RNA\n 4. Random Forest \n 5. Máquinas de soporte Vectorial\n input: ');

if Tipo==1 %Funciones discriminantes gaussianas
    for fold=1:Rept
        [Xtrain,Xtest,Ytrain,Ytest]=PartirMuestrasFold(Rept,N, X, Y, fold, Tipo);         
        [Xtrain,mu,sigma]=zscore(Xtrain);
        Xtest=normalizar(Xtest,mu,sigma);
        [Yesti,~] = classify(Xtest ,Xtrain, Ytrain,'diaglinear');%[linear diaglinear quadratic diagquadratic mahalanobis]
        NXtest = size(Xtest,1);
        MatrizConfusion = zeros(NClases,NClases);
        MatrizConfusion = MatrizYMedidasFold(MatrizConfusion, Yesti, Ytest, NXtest);  
        diagonal = diag(MatrizConfusion);
        EficienciaTest(fold) = sum(diag(MatrizConfusion))/sum(sum(MatrizConfusion));
        PrecisionTest(fold,:) = diagonal' ./(sum(MatrizConfusion'));
        SensibilidadTest(fold,:) = diagonal' ./(sum(MatrizConfusion));
    end
    IC = std(EficienciaTest);
    Error=1-mean(EficienciaTest);
    resultados(mean(EficienciaTest),Error, mean(SensibilidadTest), mean(PrecisionTest), IC);       
elseif Tipo==2 %k vecinos   
    for fold=1:Rept
        [Xtrain,Xtest,Ytrain,Ytest]=PartirMuestrasFold(Rept,N, X, Y, fold, Tipo);         
        %%% Normalización %%%

        [Xtrain,mu,sigma]=zscore(Xtrain);
        Xtest=normalizar(Xtest,mu,sigma);

        k=1;
        Yesti=vecinosCercanos(Xtest,Xtrain,Ytrain,k,'class'); 
        
        NXtest = size(Xtest,1);
        MatrizConfusion = zeros(NClases,NClases);
        MatrizConfusion = MatrizYMedidasFold(MatrizConfusion, Yesti, Ytest, NXtest);

        diagonal = diag(MatrizConfusion);
        EficienciaTest(fold) = sum(diag(MatrizConfusion))/sum(sum(MatrizConfusion));
        PrecisionTest(fold,:) = diagonal' ./(sum(MatrizConfusion'));
        SensibilidadTest(fold,:) = diagonal' ./(sum(MatrizConfusion));
                
    end
    IC = std(EficienciaTest);
    Error=1-mean(EficienciaTest);
    resultados(mean(EficienciaTest),Error, mean(SensibilidadTest), mean(PrecisionTest), IC);
elseif Tipo==3
    epocas= 50;
    neuronas = 10;
    %[~,loc]=ismember(Y,unique(Y));
    %y_one_hot = ind2vec(loc')';
    %Y=full(y_one_hot);
    net = feedforwardnet(10);
    net.trainParam.epochs = epocas;
    net.layers{1}.size=neuronas; 
    for fold=1:Rept
        [Xtrain,Xtest,Ytrain,Ytest]=PartirMuestrasFold(Rept,N, X, Y, fold, Tipo);

        %%% Normalización %%%

        [Xtrain,mu,sigma]=zscore(Xtrain);
        Xtest=normalizar(Xtest,mu,sigma);

        net = train(net,Xtrain',Ytrain');
        yest = sim(net,Xtest');
        yest = yest';
        [~,Yest] = max(yest,[],2);

        NXtest = size(Xtest,1);
        MatrizConfusion = zeros(NClases,NClases);
        MatrizConfusion = MatrizYMedidasFold(MatrizConfusion, Yest, Ytest, NXtest);
        
        diagonal = diag(MatrizConfusion);
        EficienciaTest(fold) = sum(diag(MatrizConfusion))/sum(sum(MatrizConfusion));
        PrecisionTest(fold,:) = diagonal' ./(sum(MatrizConfusion'));
        SensibilidadTest(fold,:) = diagonal' ./(sum(MatrizConfusion));
    end
    IC = std(EficienciaTest);
    Error=1-mean(EficienciaTest);
    resultados(mean(EficienciaTest),Error, mean(SensibilidadTest), mean(PrecisionTest), IC);
elseif Tipo==4
    for fold=1:Rept
        [Xtrain,Xtest,Ytrain,Ytest]=PartirMuestrasFold(Rept,N, X, Y, fold, Tipo);
        
        NumArboles=200;
        Modelo=entrenarFOREST(NumArboles,Xtrain,Ytrain);
        Yest=testFOREST(Modelo,Xtest);       

        NXtest = size(Xtest,1);
        MatrizConfusion = zeros(NClases,NClases);
        MatrizConfusion = MatrizYMedidasFold(MatrizConfusion, Yest, Ytest, NXtest);
        
        diagonal = diag(MatrizConfusion);
        EficienciaTest(fold) = sum(diag(MatrizConfusion))/sum(sum(MatrizConfusion));
        PrecisionTest(fold,:) = diagonal' ./(sum(MatrizConfusion'));
        SensibilidadTest(fold,:) = diagonal' ./(sum(MatrizConfusion));        
    end
    IC = std(EficienciaTest);
    Error=1-mean(EficienciaTest);
    resultados(mean(EficienciaTest),Error, mean(SensibilidadTest), mean(PrecisionTest), IC);
elseif Tipo==5    
    k = 'g';%kernel
    for fold=1:Rept
        [Xtrain,Xtest,Ytrain,Ytest]=PartirMuestrasFold(Rept,N, X, Y, fold, Tipo);

        [Xtrain,mu,sigma]=zscore(Xtrain);
        %Xtest=normalizar(Xtest,mu,sigma);
        Xtest=(Xtest - repmat(mu,size(Xtest,1),1))./repmat(sigma,size(Xtest,1),1);
        
        Ytrain1 = Ytrain;
        Ytrain1(Ytrain ~= 1) = -1;
        Modelo1=entrenarSVM(Xtrain,Ytrain1,'classification',boxConstraint,gamma);

        Ytrain2 = Ytrain;
        Ytrain2(Ytrain ~= 2) = -1;
        Ytrain2(Ytrain == 2) = 1;
        Modelo2=entrenarSVM(Xtrain,Ytrain2,'classification',boxConstraint,gamma);  
        
        Ytrain3 = Ytrain;
        Ytrain3(Ytrain ~= 3) = -1;
        Ytrain3(Ytrain == 3) = 1;
        Modelo3=entrenarSVM(Xtrain,Ytrain3,'classification',boxConstraint,gamma);
        
        Ytrain4 = Ytrain;
        Ytrain4(Ytrain ~= 4) = -1;
        Ytrain4(Ytrain == 4) = 1;
        Modelo4=entrenarSVM(Xtrain,Ytrain4,'classification',boxConstraint,gamma);

        [~,Yest1]=testSVM(Modelo1,Xtest);
        [~,Yest2]=testSVM(Modelo2,Xtest); 
        [~,Yest3]=testSVM(Modelo3,Xtest); 
        [~,Yest4]=testSVM(Modelo4,Xtest); 

        [~,Yest] = max([Yest1,Yest2,Yest3,Yest4],[],2); 
        
        NXtest = size(Xtest,1);
        MatrizConfusion = zeros(NClases,NClases);
        MatrizConfusion = MatrizYMedidasFold(MatrizConfusion, Yest, Ytest, NXtest);
        
        diagonal = diag(MatrizConfusion);
        EficienciaTest(fold) = sum(diag(MatrizConfusion))/sum(sum(MatrizConfusion));
        PrecisionTest(fold,:) = diagonal' ./(sum(MatrizConfusion'));
        SensibilidadTest(fold,:) = diagonal' ./(sum(MatrizConfusion));         
    end
    IC = std(EficienciaTest);
    Error=1-mean(EficienciaTest);
    resultados(mean(EficienciaTest),Error, mean(SensibilidadTest), mean(PrecisionTest), IC);
end
    
