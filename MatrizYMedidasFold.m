function [MatrizConfusion]= MatrizYMedidasFold(MatrizConfusion, Yesti, Ytest, NXtest)
    for i=1:NXtest
        MatrizConfusion(Yesti(i),Ytest(i)) = MatrizConfusion(Yesti(i),Ytest(i)) + 1;
    end                
end