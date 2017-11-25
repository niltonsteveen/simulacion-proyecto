function [Xtrain, Xtest, Ytrain, Ytest] = PartirMuestrasFold(Rept,N, X, Y, fold, tipo)
    rng('default');
    particion=cvpartition(Y,'Kfold',Rept);
    indices=particion.training(fold);
    Xtrain=X(particion.training(fold),:);
    Xtest=X(particion.test(fold),:);
    if tipo==3  
        Y2=(Y')';
        Ytest=Y2(particion.test(fold));
        [~,loc]=ismember(Y,unique(Y));
        y_one_hot = ind2vec(loc')';
        Y=full(y_one_hot);
        Ytrain=Y(particion.training(fold),:);
    else
        Ytrain=Y(particion.training(fold));
        Ytest=Y(particion.test(fold));
    end    
    
    muestrasClaseMinoritaria=Xtrain(find(Ytrain==2),:);
    muestrasSinteticas=SMOTE(muestrasClaseMinoritaria,20,5);
    muestrasClaseMinoritaria1=Xtrain(find(Ytrain==4),:);
    muestrasSinteticas1=SMOTE(muestrasClaseMinoritaria1,20,5);
    nuevasClases=ones(size(muestrasSinteticas,1),1)*2;
    nuevasClases1=ones(size(muestrasSinteticas1,1),1)*4;
    Xtrain=[Xtrain;muestrasSinteticas];
    Ytrain=[Ytrain;nuevasClases];
    Xtrain=[Xtrain;muestrasSinteticas1];
    Ytrain=[Ytrain;nuevasClases1];
end