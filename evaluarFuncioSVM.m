function Salida = evaluarFuncioSVM(Alpha,Bias,Targets,SupportVectors,Muestra,sigma,kernel)

    Alpha=abs(Alpha);
    Salida=((Alpha.*Targets)'*kernel_mat(Muestra,SupportVectors,sigma,kernel))+Bias;

end