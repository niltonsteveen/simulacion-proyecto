function [Ytest,PYTest] = testSVM(Modelo,Xtest)
 [Ytest,PYTest]= simlssvm(Modelo,Xtest);
end