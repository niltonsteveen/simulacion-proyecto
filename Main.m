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

N=size(X,1); %Numero de muestras
NClases=length(unique(Y));

Tipo=input('Ingrese según lo desee: \n 1. Funciones Discriminantes Gausianas\n 2. K-nn\n 3. RNA\n 4. Random Forest \n 5. Máquinas de soporte Vectorial\n input: ');

if Tipo==1 %Funciones discriminantes gaussianas
    for fold=1:Rept
        [Xtrain,Xtest,Ytrain,Ytest]=PartirMuestrasFold(Rept,N, X, Y, fold, Tipo);         
        [Xtrain,mu,sigma]=zscore(Xtrain);
        Xtest=normalizar(Xtest,mu,sigma);
        [Yesti,~] = classify(Xtest ,Xtrain, Ytrain,'linear');%[linear diaglinear quadratic diagquadratic mahalanobis]
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

        k=100;
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
    epocas= 800;
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
        
        NumArboles=500;
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
    boxConstraint=0.01; % Este es el parametro de regularización
    gamma=0.01; % Este es el parametro (si lo necesita) de la funcion kernel
    tipoK=1; % 1 = Lineal, 2 = Gaussiano(o RBF)

    TipoValidacion=input('Ingrese 1 para validacion Bootstrap ó 2 para validacion cruzada\n input:');   
    if TipoValidacion==1 
        rng('default');
        ind=randperm(N);
        
        Xtrain=X(ind(1:3819),:);
        Xtest=X(ind(3820:end),:);
        Ytrain=Y(ind(1:3819),:);
        Ytest=Y(ind(3820:end),:);
        
        [Xtrain,mu,sigma]=zscore(Xtrain);
        Xtest=normalizar(Xtest,mu,sigma);
        
        Ytrain1 = Ytrain; 
        Ytrain1(Ytrain1~=1)=-1;
        Modelo1 = entrenarSVM(Xtrain,Ytrain1,'c',boxConstraint,gamma,tipoK);
        alpha1 = Modelo1.alpha;
        b1 = Modelo1.b;

        Ytrain2 = Ytrain;
        Ytrain2(Ytrain2~=2)=-1; 
        %Ytrain2(Ytrain2==2)=1; 
        Modelo2 = entrenarSVM(Xtrain,Ytrain2,'c',boxConstraint,gamma,tipoK); 
        alpha2 = Modelo2.alpha;
        b2 = Modelo2.b;
        
        Ytrain3 = Ytrain; 
        Ytrain3(Ytrain3~=3)=-1; 
        %Ytrain3(Ytrain3==3)=1; 
        Modelo3 = entrenarSVM(Xtrain,Ytrain3,'c',boxConstraint,gamma,tipoK);
        alpha3 = Modelo3.alpha;
        b3 = Modelo3.b;
        
        Ytrain4 = Ytrain; 
        Ytrain4(Ytrain4~=4)=-1; 
        %Ytrain3(Ytrain3==3)=1; 
        Modelo4 = entrenarSVM(Xtrain,Ytrain4,'c',boxConstraint,gamma,tipoK);
        alpha4 = Modelo4.alpha;
        b4 = Modelo4.b;
        
        [Yest1,YestContinuo1]=testSVM(Modelo1,Xtest);
        [Yest2,YestContinuo2]=testSVM(Modelo2,Xtest);
        [Yest3,YestContinuo3]=testSVM(Modelo3,Xtest);
        [Yest4,YestContinuo4]=testSVM(Modelo4,Xtest);
        
        if (tipoK == 1)
            K = kernel_matrix(Xtrain, 'lin_kernel', [], Xtest);
        elseif (tipoK == 2)
            K = kernel_matrix(Xtrain, 'RBF_kernel', gamma, Xtest);
        end

        Ytemp1 = (alpha1'*K + b1)';
        Ytemp2 = (alpha2'*K + b2)';
        Ytemp3 = (alpha3'*K + b3)';
        Ytemp4 = (alpha4'*K + b4)';

        YestContinuo=[YestContinuo1,YestContinuo2,YestContinuo3,YestContinuo4];
        Ytemp=[Ytemp1,Ytemp2,Ytemp3,Ytemp4];
        
        [~,Yest]=max(YestContinuo,[],2); 
        [~,Yesti]=max(Ytemp,[],2); 
        
        Eficiencia=(sum(Yest==Ytest))/length(Ytest);
        Error=1-Eficiencia;

        Texto=strcat('La eficiencia en prueba es: ',{' '},num2str(Eficiencia));
        disp(Texto);
        Texto=strcat('El error de clasificación en prueba es: ',{' '},num2str(Error));
        disp(Texto);        
    elseif TipoValidacion==2
        NumMuestras=size(X,1); 
        EficienciaTest=zeros(1,Rept);
        for fold=1:Rept
            rng('default');
            particion=cvpartition(NumMuestras,'Kfold',Rept);
            indices=particion.training(fold);
            Xtrain=X(particion.training(fold),:);
            Xtest=X(particion.test(fold),:);
            Ytrain=Y(particion.training(fold));
            Ytest=Y(particion.test(fold));
            
            [Xtrain,mu,sigma]=zscore(Xtrain);
            Xtest=normalizar(Xtest,mu,sigma);
            
            Ytrain1 = Ytrain; 
            Ytrain1(Ytrain1~=1)=-1;
            Modelo1 = entrenarSVM(Xtrain,Ytrain1,'c',boxConstraint,gamma,tipoK);
            alpha1 = Modelo1.alpha;
            b1 = Modelo1.b;

            Ytrain2 = Ytrain;
            Ytrain2(Ytrain2~=2)=-1; 
            %Ytrain2(Ytrain2==2)=1; 
            Modelo2 = entrenarSVM(Xtrain,Ytrain2,'c',boxConstraint,gamma,tipoK); 
            alpha2 = Modelo2.alpha;
            b2 = Modelo2.b;

            Ytrain3 = Ytrain; 
            Ytrain3(Ytrain3~=3)=-1; 
            %Ytrain3(Ytrain3==3)=1; 
            Modelo3 = entrenarSVM(Xtrain,Ytrain3,'c',boxConstraint,gamma,tipoK);
            alpha3 = Modelo3.alpha;
            b3 = Modelo3.b;

            Ytrain4 = Ytrain; 
            Ytrain4(Ytrain4~=4)=-1; 
            %Ytrain3(Ytrain3==3)=1; 
            Modelo4 = entrenarSVM(Xtrain,Ytrain4,'c',boxConstraint,gamma,tipoK);
            alpha4 = Modelo4.alpha;
            b4 = Modelo4.b;

            [Yest1,YestContinuo1]=testSVM(Modelo1,Xtest);
            [Yest2,YestContinuo2]=testSVM(Modelo2,Xtest);
            [Yest3,YestContinuo3]=testSVM(Modelo3,Xtest);
            [Yest4,YestContinuo4]=testSVM(Modelo4,Xtest);

            if (tipoK == 1)
                K = kernel_matrix(Xtrain, 'lin_kernel', [], Xtest);
            elseif (tipoK == 2)
                K = kernel_matrix(Xtrain, 'RBF_kernel', gamma, Xtest);
            end

            Ytemp1 = (alpha1'*K + b1)';
            Ytemp2 = (alpha2'*K + b2)';
            Ytemp3 = (alpha3'*K + b3)';
            Ytemp4 = (alpha4'*K + b4)';

            YestContinuo=[YestContinuo1,YestContinuo2,YestContinuo3,YestContinuo4];
            Ytemp=[Ytemp1,Ytemp2,Ytemp3,Ytemp4];

            [~,Yest]=max(YestContinuo,[],2); 
            [~,Yesti]=max(Ytemp,[],2);
            
            MatrizConfusion=zeros(NClases,NClases);
            for i=1:size(Xtest,1)
                MatrizConfusion(Yesti(i),Ytest(i))=MatrizConfusion(Yesti(i),Ytest(i)) + 1;
            end
            EficienciaTest(fold)=sum(diag(MatrizConfusion))/sum(sum(MatrizConfusion));        
        end
        Eficiencia = mean(EficienciaTest);
        Error=1-Eficiencia;

        Texto=strcat('La eficiencia en prueba es: ',{' '},num2str(Eficiencia));
        disp(Texto);
        Texto=strcat('El error de clasificación en prueba es: ',{' '},num2str(Error));
        disp(Texto); 
    end
end
    
