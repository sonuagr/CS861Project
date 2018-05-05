load('mnistTrainImages.mat');
load('mnistTrainLabels.mat');


L = 5;
numSamples =length(labels);
rep = ones(1,numSamples);

W_o = cell(L,1);
W_o{1} = csvread('W1_val_tanh.csv')';
W_o{2} = csvread('W2_val_tanh.csv')';
W_o{3} = csvread('W3_val_tanh.csv')';
W_o{4} = csvread('W4_val_tanh.csv')';
W_o{5} = csvread('W5_val_tanh.csv')';


b_o = cell(L,1);
b_o{1} = kron(csvread('b1_val_tanh.csv'), rep);
b_o{2} = kron(csvread('b2_val_tanh.csv'), rep);
b_o{3} = kron(csvread('b3_val_tanh.csv'), rep);
b_o{4} = kron(csvread('b4_val_tanh.csv'), rep);
b_o{5} = kron(csvread('b5_val_tanh.csv'), rep);


X_o = cell(L+1,1);
X_o{1} = images;
for i = 1:L-1
    X_o{i+1} = tansig(W_o{i}*X_o{i} + b_o{i});
end
X_o{L+1} = W_o{L}*X_o{L} + b_o{L};

[out,outIndex] = max(X_o{L+1});
outIndex = outIndex' - 1;


W_c = cell(L,1);
b_c = cell(L,1);
b_c_ToSave = cell(L,1);
X_c = cell(L+1,1);
X_c{1} = X_o{1};
for i = 1:L-1
    o = size(X_o{i+1});
    o = o(1);
    c = ceil(o/2);
    [U,S,V] = svds(X_o{i+1}, c);
    sigInvV = atanh(V');
    
    X_temp = [X_c{i}; ones(1,numSamples)];
    W_temp = sigInvV * X_temp' * pinv(X_temp*X_temp');
    W_c{i} =  W_temp(:,1:end-1);
    b_c{i} =  W_temp(:,end);
    b_c_ToSave{i} = b_c{i};
    b_c{i} = kron(b_c{i}, rep);
    X_c{i+1} = tansig(W_c{i}*X_c{i} + b_c{i});
end

X_temp = [X_c{L}; ones(1,numSamples)];
W_temp = X_o{L+1} * X_temp' * pinv(X_temp*X_temp')/3000;
W_c{L} =  W_temp(:,1:end-1);
b_c{L} =  W_temp(:,end);
b_c_ToSave{L} = b_c{L};
b_c{L} = kron(b_c{L}, rep);
X_c{L+1} = W_c{L}*X_c{L} + b_c{L};

[outc,outIndexc] = max(X_c{L+1});
outIndexc = outIndexc' - 1;

diff_o = nnz(labels-outIndex);
diff_c = nnz(labels-outIndexc);
diff_oc = nnz(outIndexc-outIndex);

norm_o = [norm(W_o{1},'fro'), norm(W_o{2},'fro'), norm(W_o{3},'fro'), norm(W_o{4},'fro'), norm(W_o{5},'fro')];
norm_c = [norm(W_c{1},'fro'), norm(W_c{2},'fro'), norm(W_c{3},'fro'), norm(W_c{4},'fro'), norm(W_c{5},'fro')];


b_c = b_c_ToSave;

save('compressedWeights.mat','W_c');
save('compressedBiases.mat','b_c');
save('compressedX.mat','X_c');
save('originalX.mat','X_o');
