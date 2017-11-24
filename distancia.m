function distances = distancia(Xtrain, Xval)
    N = size(Xtrain, 1);
    distances = zeros(N, 1);

    for i = 1:N
        temp = (Xtrain(i, :) - Xval) .^ 2;
        distances(i) = sqrt(sum(temp, 2));
    end