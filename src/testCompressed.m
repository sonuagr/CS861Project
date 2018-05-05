load('compressedWeights.mat');
load('compressedBiases.mat');


load('mnistTestImages.mat');
load('mnistTestLabels.mat');

L = 5;

numSamples =length(labels);
rep = ones(1,numSamples);

X_c = cell(L+1,1);
X_c{1} = images;

for i = 1:L-1
    b_c{i} = kron(b_c{i}, rep);
    X_c{i+1} = tansig(W_c{i}*X_c{i} + b_c{i});
end

X_c{L+1} = W_c{L}*X_c{L} + b_c{L};

[outc,outIndexc] = max(X_c{L+1});
outIndexc = outIndexc' - 1;

diff_c = nnz(labels-outIndexc);