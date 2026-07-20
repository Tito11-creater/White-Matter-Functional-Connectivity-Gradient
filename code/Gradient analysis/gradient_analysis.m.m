function gradient_analysis()

% Compute functional connectivity gradients from group-average FC
% 
% This function loads a group-averaged functional connectivity matrix,
% applies thresholding at multiple sparsity levels, constructs similarity
% matrices using a Gaussian kernel, averages them, and computes
% connectome gradients via the BrainSpace toolbox.
%
% Outputs:
%   - Thresholded FC matrices (FC_sparse_*.txt)
%   - Affinity matrices per sparsity (aff_sparse_*.txt)
%   - Average affinity matrix (aff_average.txt)
%   - Gradient components (gradient_test_group.txt)
%   - Explained variance (lambda_test_group.txt)

%% ====================== Load data ======================================
% Change to the directory containing the results
cd('/data/FC/');

% Load group-averaged FC matrix (200x200)
FC_test = load('FC_test_group_mean_r.txt');

%% ====================== Parameter settings =============================
nNodes = 200;                % Number of brain regions
nSparsity = 11;              % Number of sparsity levels (0.1 to 0.2, step 0.01)
sparsityStart = 0.1;
sparsityStep = 0.01;

% Initialize accumulator for averaged affinity matrix
sim_average = zeros(nNodes, nNodes);

%% ====================== Thresholding and affinity computation ==========
for i = 1:nSparsity
    % Current sparsity level
    sparse = sparsityStart + (i - 1) * sparsityStep;
    
    % ---------- Threshold the FC matrix --------------------------------
    FC_thre = zeros(nNodes, nNodes);
    for row = 1:nNodes
        FC_row = FC_test(row, :);
        FC_sort = sort(FC_row);
        % Threshold value that keeps the top (1-sparse) fraction of edges
        thre = FC_sort(round(nNodes * (1 - sparse)));
        FC_row(FC_row <= thre) = 0;
        FC_thre(row, :) = FC_row;
    end
    % Save thresholded matrix
    writematrix(FC_thre, ['FC_sparse_', num2str(sparse), '.txt']);
    
    % ---------- Build affinity matrix using Gaussian kernel ------------
    sim = zeros(nNodes, nNodes);
    for ii = 1:nNodes
        for jj = ii:nNodes
            % Euclidean distance between rows ii and jj, then Gaussian kernel
            % sigma^2 = nNodes (200) as per original code
            dist = norm(FC_thre(ii, :) - FC_thre(jj, :), 2);
            sim(ii, jj) = exp(-(dist^2) / nNodes);
            sim(jj, ii) = sim(ii, jj);
        end
    end
    % Save affinity matrix for this sparsity
    writematrix(sim, ['aff_sparse_', num2str(sparse), '.txt']);
    
    % Accumulate for averaging
    sim_average = sim_average + sim;
end

% Average across all sparsity levels
sim_average = sim_average / nSparsity;
writematrix(sim_average, 'aff_average.txt');

%% ====================== Gradient computation ===========================
% Use the BrainSpace toolbox (GradientMaps)
% Kernel: none (directly on affinity), approach: diffusion maps, 1 component
G_test = GradientMaps('kernel', 'none', 'approach', 'dm', 'n_components', 1);
G_test = G_test.fit(sim_average, 'sparsity', 0);

% Extract gradients
G1 = G_test.gradients{1}(:, 1);

%% ====================== Save outputs ===================================
% Save gradient component
writematrix(G1, 'gradient_test_group.txt');
% Save explained variance (normalized eigenvalues)
writematrix(G_test.lambda{1} / sum(G_test.lambda{1}), 'lambda_test_group.txt');

end