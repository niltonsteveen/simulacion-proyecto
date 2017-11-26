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

Tipo=input('Ingrese según lo desee: \n 1. Funciones Discriminantes Gausianas\n 2. K-nn\n 3. RNA\n 4. Random Forest \n 5. Máquinas de soporte Vectorial\n 6. Fisher \n 7. Simulaciones con los 3 mejores input: ');

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
        
        NumArboles=10;
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
elseif Tipo==6   
    % Se normalizan los datos   
	X = zscore(X);
    % NOTA: Ya revise y no hace la diferencia si se normaliza o no
    
    %%%%%%%%%%%% Correlacion entre las caracteristicas X
    
    alpha = 0.05; % Use 0.05 para un IC del 95% [Opcion por defecto] | Use 0.1 para un IC del 90% | etc.....
    [correlacion,p]= corrcoef([X,Y],'alpha',alpha); % Se calcula la matriz de coeficientes de correlacion y valores p
    
    % Luego se grafican dichas matrices
    figure(1)
    plottable(correlacion);
    title('Matriz De Correlacion X');
    
    figure(2)
    imagesc(correlacion);
    colorbar;
    
    figure(3)
    plottable(p);
    title('Matriz De Valores P');
    %NOTA1: Si el valor de p es más pequeño que el valor de alpha entonces hay una correlacion significativa
    
    %NOTA2: Recuerde que lo ideal es:
    %                   Poca correlacion entre las X porque ello me indica
    %                   que las caracteristicas son independientes y cada
    %                   una aporta informacion.
    %                   Mucha correlacion entre las X y la Y porque ello indica
    %                   que si hay algun vinculo o relacion entre las
    %                   muestras y la salida esperada.
    
    %%%%%%%%%%%% Discriminante de Fisher
    
    % 1ero - Se determinan las posiciones de las muestra de cada clase
    indicesClase1 = find(Y == 1);
    indicesClase2 = find(Y == 2);
    indicesClase3 = find(Y == 3);
    indicesClase4 = find(Y == 4);
    
    % 2do - Se calculan las medias de cada clase (o sea, por columnas) y se
    %           unen en un solo arreglo
    mediaClase1 = mean(X(indicesClase1,:) ,1);
    mediaClase2 = mean(X(indicesClase2,:) ,1);
    mediaClase3 = mean(X(indicesClase3,:) ,1);
    mediaClase4 = mean(X(indicesClase4,:) ,1);
    
    media = [mediaClase1; mediaClase2; mediaClase3; mediaClase4];
    
    % 3ero - Se calculan las varianzas de cada clase (o sea, por columnas) y se
    %           unen en un solo arreglo
    varClase1 = var(X(indicesClase1,:) ,1);
    varClase2 = var(X(indicesClase2,:) ,1);
    varClase3 = var(X(indicesClase3,:) ,1);
    varClase4 = var(X(indicesClase4,:) ,1);
    
    varianza = [varClase1;varClase2;varClase3;varClase4];
    
    % 4to - Se calcula el indice de Fisher
    
    coef = zeros(1,24);
    
    for i=1:4
        for j=1:4
            if (j ~= i)
                numerador = (media(i,:) - media(j,:)).^2;
                denominador = varianza(i,:) + varianza(j,:);
                coef = coef + (numerador./denominador);
            end
        end
    end
    
    % y 5to - Se muestra el resultado tanto sin procesar como normalizado
    Texto = ['Indice de Fisher: ', num2str(coef)];
    disp(Texto);
    
    coefN = coef./max(coef);
    Texto = ['Indice de Fisher Normalizado: ', num2str(coefN)];
    disp(Texto);
    
    % Se hace un dibujo de los indices normalizados para mayor claridad
    figure(4)
    ejeX = 1:24;
    stem(ejeX, coefN);
    title('Indices Fisher Normalizados');
elseif Tipo==7
    opciones = statset('display','iter'); 
    sentido = 'forward'; 
    
    [caracteristicasElegidas, proceso] = sequentialfs(@funcionForest,X,Y,'direction',sentido,'options',opciones);
    
    X = X(:, caracteristicasElegidas);
    
   % TipoModelo=input('Ingrese 1 para random forest, 2 para knn y 3 para rna \n input:');
    for fold=1:Rept
        [Xtrain,Xtest,Ytrain,Ytest]=PartirMuestrasFold(Rept,N, X, Y, fold, 2);

        [Xtrain,mu,sigma] = zscore(Xtrain);
        Xtest = (Xtest - repmat(mu,size(Xtest,1),1))./repmat(sigma,size(Xtest,1),1);

        NumArboles=100;
        Modelo = TreeBagger(NumArboles,Xtrain,Ytrain,'oobvarimp', 'On');

        % Se obtienen las predicciones del modelo con base al modelo
        % entrenado y las muestras separadas para la validacion del sistema
        Yest = predict(Modelo,Xtest);
        Yest = str2double(Yest);

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
    
    Rept=10;
    EficienciaTest=zeros(1,Rept);
    PrecisionTest=zeros(Rept,4);
    SensibilidadTest=zeros(Rept,4);
    
    for fold=1:Rept
        [Xtrain,Xtest,Ytrain,Ytest]=PartirMuestrasFold(Rept,N, X, Y, fold, 1);         
        %%% Normalización %%%

        [Xtrain,mu,sigma]=zscore(Xtrain);
        Xtest=normalizar(Xtest,mu,sigma);

        k=1;
        Yesti=vecinosCercanos(Xtest,Xtrain,Ytrain,k,'class'); 

        NXtest = size(Xtest,1);
        MatrizConfusion = zeros(NClases,NClases);
        Matriz5Confusion = MatrizYMedidasFold(MatrizConfusion, Yesti, Ytest, NXtest);

        diagonal = diag(MatrizConfusion);
        EficienciaTest(fold) = sum(diag(MatrizConfusion))/sum(sum(MatrizConfusion));
        PrecisionTest(fold,:) = diagonal' ./(sum(MatrizConfusion'));
        SensibilidadTest(fold,:) = diagonal' ./(sum(MatrizConfusion));

    end
    IC = std(EficienciaTest);
    Error=1-mean(EficienciaTest);
    resultados(mean(EficienciaTest),Error, mean(SensibilidadTest), mean(PrecisionTest), IC);

    
    epocas= 50;
    neuronas = 60;
    %[~,loc]=ismember(Y,unique(Y));
    %y_one_hot = ind2vec(loc')';
    %Y=full(y_one_hot);
    net = feedforwardnet(10);
    net.trainParam.epochs = epocas;
    net.layers{1}.size=neuronas; 
    Rept=10;
    EficienciaTest=zeros(1,Rept);
    PrecisionTest=zeros(Rept,4);
    SensibilidadTest=zeros(Rept,4);
    for fold=1:Rept
        [Xtrain,Xtest,Ytrain,Ytest]=PartirMuestrasFold(Rept,N, X, Y, fold, 3);

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
end
    
