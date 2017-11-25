function Modelo = entrenarSVM(X,Y,tipo,boxConstraint,sigma)
    %Modelo = trainlssvm({X,Y, tipo, boxConstraint,sigma,'lin_kernel'});
    Modelo = trainlssvm({X,Y, tipo, boxConstraint,sigma});
end