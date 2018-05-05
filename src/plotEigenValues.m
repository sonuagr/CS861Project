load('originalX.mat');

L=5;

for i = 3:L-1
    o = size(X_o{i+1});
    o = o(1);
%     c = ceil(o/2);
    [U,S,V] = svds(X_o{i+1}, o);  
    figure(i);
    plot(diag(S));
    title('Plot of singular values correspnding to SVD of X^4')
    ylabel('Singular values')
    xlabel('Indices')
end
    