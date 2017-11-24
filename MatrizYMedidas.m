function [Eficiencia, Sensibilidad, Precision, Error]=MatrizYMedidas(MatrizConfusion, Yesti, Ytest, NXtest)
    for i=1:NXtest
        MatrizConfusion(Yesti(i),Ytest(i)) = MatrizConfusion(Yesti(i),Ytest(i)) + 1;
    end
    diagonal = diag(MatrizConfusion);
    Eficiencia = sum(diag(MatrizConfusion))/sum(sum(MatrizConfusion));
    Sensibilidad = diagonal' ./(sum(MatrizConfusion));
    Precision = diagonal' ./(sum(MatrizConfusion'));     
    Error=1-Eficiencia;
end