function Yesti = vecinosCercanos(Xval,Xent,Yent,k,tipo)

    %%% El parametro 'tipo' es el tipo de problema que se va a resolver
    
    %%% La función debe retornar el valor de predicción Yesti para cada una de 
    %%% las muestras en Xval. Por esa razón Yesti se inicializa como un vectores 
    %%% de ceros, de dimensión M.

    N=size(Xent,1); %N es el numero de muestras de entrenamiento
    M=size(Xval,1); %M es el numero de muestras de validacion
    
    Yesti=zeros(M,1);
    dis=zeros(N,1);

    if strcmp(tipo,'class')
        
        for j=1:M
            %%% Complete el codigo %%%
             dis=distancia(Xent, Xval(j,:));           
             [dis, sortIndexes]= sort(dis);
             temp = Yent(sortIndexes(1:k));
             Yesti(j)= mode (temp);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%
            
        end
        
        
    elseif strcmp(tipo,'regress')
        
        for j=1:M
            %%% Complete el codigo %%%
			dis=distancia(Xent, Xval(j,:));
            [dis, sortIndexes]= sort(dis);    
            temp = Yent(sortIndexes(1:k));
            Yesti(j)= mean (temp);
			%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
        end

        
    end

end