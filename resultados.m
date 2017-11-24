function  resultados(Eficiencia,Error, Sensibilidad, Precision, IC)    
    disp(strcat('La eficiencia obtenida fue = ', num2str(Eficiencia),' +- ',num2str(IC)));
    disp(strcat('El error de clasificación en prueba es: ',{' '},num2str(Error)));
    disp(strcat('La precicion es: ',{' '},num2str(Precision)));
    disp(strcat('La sensibilidad es: ',{' '},num2str(Sensibilidad)));
end









