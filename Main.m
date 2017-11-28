clc
clear all
% close all
load('sensor_data.csv');
%Se cargan las muestras
X=sensor_data(1:end,1:end-1); 
X2=X(:, 12);
X2=[X2, X(:, 14)];
X2=[X2, X(:, 15)];
X2=[X2, X(:, 18)];
X2=[X2, X(:, 19)];
X2=[X2, X(:, 20)];
%X = X(:, caracteristicasElegidas);
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

Tipo=input('Ingrese según lo desee: \n 1. Funciones Discriminantes Gausianas\n 2. K-nn\n 3. RNA\n 4. Random Forest \n 5. Máquinas de soporte Vectorial\n 6. Fisher \n 7. Simulaciones con los 3 mejores\n 8. PCA \n input: ');

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
        [Xtrain,Xtest,Ytrain,Ytest]=PartirMuestrasFold(Rept,N, X2, Y, fold, Tipo);         
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
    neuronas = 22;
    %[~,loc]=ismember(Y,unique(Y));
    %y_one_hot = ind2vec(loc')';
    %Y=full(y_one_hot);
    net = feedforwardnet(10);
    net.trainParam.epochs = epocas;
    net.layers{1}.size=neuronas; 
    for fold=1:Rept
        [Xtrain,Xtest,Ytrain,Ytest]=PartirMuestrasFold(Rept,N, X2, Y, fold, Tipo);

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
        [Xtrain,Xtest,Ytrain,Ytest]=PartirMuestrasFold(Rept,N, X2, Y, fold, Tipo);
        
        NumArboles=100;
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
        [Xtrain,Xtest,Ytrain,Ytest]=PartirMuestrasFold(Rept,N, X2, Y, fold, Tipo);

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
elseif Tipo==8
    umbralPorcentajeDeVarianza = 85;
    epocas= 100;
    neuronas = 60;
    %[~,loc]=ismember(Y,unique(Y));
    %y_one_hot = ind2vec(loc')';
    %Y=full(y_one_hot);
    net = feedforwardnet(10);
    net.trainParam.epochs = epocas;
    net.layers{1}.size=neuronas;
    for fold=1:Rept
        [Xtrain,Xtest,Ytrain,Ytest]=PartirMuestrasFold(Rept,N, X, Y, fold, 3);
        
        %%% Se normalizan los datos %%%

        [Xtrain,mu,sigma] = zscore(Xtrain);
        Xtest = (Xtest - repmat(mu,size(Xtest,1),1))./repmat(sigma,size(Xtest,1),1);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % Ahora, se extraen los componentes principales, de modo que:
        
        % Se usa la función PCA de matalab para obtener los coeficientes de los componentes principales, los scores, las
        % varianzas de los componentes principales y el porcentaje de varianza explicada de estos. La sumatoria de este ultimo
        % retorno debe dar un total del 100%
        [coefCompPrincipales,scores,covarianzaEigenValores,~,porcentajeVarianzaExplicada,~] = pca(Xtrain);
        
        % A continuacion, se almacena el numero original de variables que tiene el sistema
        numVariables = length(covarianzaEigenValores);
        % También, se crea un variable con la cual se guardara el numero de componentes principales cuyos porcentajes de varianza sumada superan el porcentaje de varianza limite deseada
        numCompAdmitidos = 0;
        
        % Luego, se crean unas variables que almacenaran coordenadas de unas graficas que se dibujaran más adelante
        porcentajeVarianzaAcumulada = zeros(numVariables,1);
        puntosUmbral = ones(numVariables,1)*umbralPorcentajeDeVarianza;
        ejeComponentes = 1:numVariables;
        
        % PARA k que comienza en 1 HASTA el numero original de componentes HAGA
        for k=1:numVariables
            % Sume la varianza de los componentes 1 hasta k y guadelo en porcentajeVarianzaAcumulada(k)
            porcentajeVarianzaAcumulada(k) = sum(porcentajeVarianzaExplicada(1:k));
            
            %porcentajeVarianzaAcumulada(k) = sum(covarianzaEigenValores(1:k)) ./ sum(covarianzaEigenValores); % Otra forma de hacer la instruccion anterior pero los valores quedan entre 0 y 1.
            
            % SI la suma de los k componentes supera el limite de varianza deseado Y todavia no se ha establecido un numero de componentes a dejar para el sistema ENTONCES
            if (sum(porcentajeVarianzaExplicada(1:k)) >= umbralPorcentajeDeVarianza) && (numCompAdmitidos == 0)
                numCompAdmitidos = k; % Se guarda el numero de la iteracion puesto que este es el numero de componentes a tener en cuenta para el sistema
            end
        end
        
        % Una vez se calculan los varianzas acumuladas, se dibujan dos graficas:
        
        % La primera es una grafica de la magnitud de los EigenValores
        figure(1)
        stem(ejeComponentes, covarianzaEigenValores)
        xlim([1 numVariables]);
        title('Varianza de los componentes principales');
        xlabel('Componentes principales');
        ylabel('EigenValor');
        
        % La segunda grafica consiste en la acumulacion progresiva de la varianza a medida que se recorren los componentes y cual es el limite o umbral de varianza acumulada que se fijo para incluir el numero de componentes principales.
        figure(2)
        plot(ejeComponentes, porcentajeVarianzaAcumulada);
        xlim([1 numVariables]);
        hold on;
        plot(ejeComponentes, puntosUmbral,'r');
        title('Varianza acumulada de los componentes principales');
        xlabel('Componentes principales');
        ylabel('Varianza explicada (%)');
        hold off;
        
        % Ya determinado el numero de componentes con los que se quiere trabajar se estiman o proyectan los datos sobre dichos componentes principales para que el sistema trabaje con ellos
        aux = Xtrain*coefCompPrincipales;
        Xtrain = aux(:,1:numCompAdmitidos);
        
        aux = Xtest*coefCompPrincipales;
        Xtest = aux(:,1:numCompAdmitidos);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % Se hace el entrenamiento del modelo
        net = train(net,Xtrain',Ytrain');
        yest = sim(net,Xtest');
        yest = yest';
        [~,Yest] = max(yest,[],2); 
        % Por ultimo, se calcula la eficiencia de esta iteracion
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
    
