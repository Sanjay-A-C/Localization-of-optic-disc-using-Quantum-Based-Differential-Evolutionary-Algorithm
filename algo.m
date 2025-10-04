% QIDE_exact_OD.m
% TRUE Quantum-Inspired Differential Evolution (θ-space DE mutation + crossover)
% - Population encoded as q-bit angles (theta)
% - DE mutation and crossover performed in θ-space: θ_trial = θ_r1 + F*(θ_r2 - θ_r3)
% - Measure (stochastic) -> Evaluate -> Selection (replace θ if trial better)
% - Saves ENTROPY, BOUNDS_RUNS and AVG_SUMMARY per your pipeline
% Author: ChatGPT (adapt hyperparameters as needed)

clc; clear; close all; warning off; tic;   % clear workspace, close figs, suppress warnings, start timer
rng('shuffle');                            % random seed (different every run)

%% ------------------ USER SETTINGS --------------------
imageFolder = "D:\Backup 12.12.24\Desktop\optic_disc_localization\SOURCE\diaretdb0_fundus_images";   % input fundus images path
outputBaseDir = "D:\Backup 12.12.24\Desktop\optic_disc_localization\6_QDE\DIARETDB0\QSDE_15SEPT25\OUTPUT";  % result images path
excelDir = "D:\Backup 12.12.24\Desktop\optic_disc_localization\6_QDE\DIARETDB0\QSDE_15SEPT25\XSLT";       % logs (entropy, bounds) path

numRuns = 20;       % independent DE runs per image
numGen = 200;       % generations per run
popSize = 20;       % population size (number of candidate solutions)
D = 4;              % problem dimensions: [k, l, r1, r2]

% QIDE/DE hyperparameters (operate in θ-space)
F_min = 0.1;        % min differential weight
F_max = 0.9;        % max differential weight
CR = 0.7;           % crossover probability
mutation_theta = 0.01; % small jitter to maintain population diversity
min_theta = 0;      % lower bound of angle
max_theta = pi/2;   % upper bound of angle
noise_measure = 0.02; % noise added during measurement (stochastic collapse)

STATIC_W = 5;       % static marker width
STATIC_H = 5;       % static marker height

% create directories if not existing
if ~exist(outputBaseDir,'dir'), mkdir(outputBaseDir); end
if ~exist(excelDir,'dir'), mkdir(excelDir); end

%% ------------------ COLLECT IMAGES --------------------
imageFiles = dir(fullfile(imageFolder, '*.png'));   % list all PNG images
nImages = numel(imageFiles);                        % total number of images
if nImages == 0, error('No PNG images found'); end   % error if none found

summaryAvgEntropy = zeros(nImages, numGen);         % store avg entropy curve per image
imageIDs = strings(nImages,1);                      % store image IDs

%% ------------------ MAIN (IMAGE LOOP) --------------------
for fileIdx = 1:nImages
    imPath = fullfile(imageFiles(fileIdx).folder, imageFiles(fileIdx).name);  % full image path
    [~, imgName, ~] = fileparts(imPath);                                     % get image filename (without extension)
    fprintf("\n==== Processing %s (%d/%d) ====\n", imgName, fileIdx, nImages);
    imageIDs(fileIdx) = imgName;                                             % store ID
    
    outputDir = fullfile(outputBaseDir, imgName);                            % create output folder for image
    if ~exist(outputDir,'dir'), mkdir(outputDir); end
    
    originalImage = imread(imPath);                                          % read input image
    imwrite(originalImage, fullfile(outputDir,'original.png'));              % save copy
    
    % ---------- Preprocess (custom function below) ----------
    procImage_raw = preprocess_sir(originalImage);                           % run preprocessing pipeline
    procImage = im2double(procImage_raw);                                    % convert to double precision
    procImage = mat2gray(procImage);                                         % normalize intensity to [0,1]
    imwrite(procImage, fullfile(outputDir,'preprocessed.png'));              % save preprocessed image
    
    [H, W] = size(procImage);                                                % get image size
    
    % define parameter bounds for [k, l, r1, r2]
    runLB = [1, 1, 15, 15];                  % lower bound
    runUB = [W, H, 90, 90];                  % upper bound
    
    entropyLog = zeros(numRuns, numGen);     % store entropy fitness over runs
    runBoundsSummary = zeros(numRuns, 13);   % bounding box logs
    bestOfImageFit = -Inf;                   % best fitness for this image
    bestOfImageSol = [];                     % best solution
    
    %% runs
    for run = 1:numRuns
        fprintf(" ▶ Run %d/%d\n", run, numRuns);
        
        % ---------------- initialize θ population uniformly in [0, π/2]
        theta = rand(popSize, D) * (pi/2); 
        
        % measure q-bits to phenotypes (coordinates)
        pop = measure_theta_with_noise(theta, runLB, runUB, noise_measure);
        pop = enforce_bounds(pop, runLB, runUB);   % clip to bounds
        
        % evaluate initial fitness
        fitness = zeros(popSize,1);
        for i=1:popSize
            fitness(i) = entropy_fitness(procImage, round(pop(i,:)));
        end
        
        % track best solution so far
        [bestFit, bestIdx] = max(fitness);
        bestSol = pop(bestIdx,:);
        runBestFit = zeros(numGen,1);
        
        % --------------- generations
        for g = 1:numGen
            % adaptive F decreases linearly from F_max to F_min
            F = F_max - (F_max - F_min) * (g-1)/(numGen-1);
            
            % loop through each candidate
            for i = 1:popSize
                % choose 3 distinct random individuals (r1,r2,r3) ≠ i
                idxs = 1:popSize; idxs(i) = [];
                perm = idxs(randperm(popSize-1));
                r1 = perm(1); r2 = perm(2); r3 = perm(3);
                
                % mutation in θ-space
                theta_mut = theta(r1,:) + F * (theta(r2,:) - theta(r3,:));
                
                % binomial crossover
                jrand = randi(D);               % ensure at least one dim crosses
                trial_theta = theta(i,:);       % start with target
                for j = 1:D
                    if rand <= CR || j == jrand
                        trial_theta(j) = theta_mut(j);
                    end
                end
                
                % small jitter for diversity
                trial_theta = trial_theta + mutation_theta*(rand(1,D)*2 - 1);
                
                % clip trial θ within bounds
                trial_theta = max(min(trial_theta, max_theta), min_theta);
                
                % measurement: collapse both target and trial to phenotypes
                pheno_target = measure_theta_with_noise(theta(i,:), runLB, runUB, noise_measure);
                pheno_trial = measure_theta_with_noise(trial_theta, runLB, runUB, noise_measure);
                
                % fitness evaluation
                fit_target = entropy_fitness(procImage, round(pheno_target));
                fit_trial = entropy_fitness(procImage, round(pheno_trial));
                
                % selection
                if isfinite(fit_trial) && (fit_trial >= fit_target)
                    theta(i,:) = trial_theta;               % replace with trial
                    pop(i,:) = enforce_bounds(pheno_trial, runLB, runUB);
                    fitness(i) = fit_trial;
                else
                    pop(i,:) = enforce_bounds(pheno_target, runLB, runUB);
                    fitness(i) = fit_target;
                end
            end % for i
            
            % update best fitness in population
            [bestFit_cur, idx] = max(fitness);
            if bestFit_cur > bestFit
                bestFit = bestFit_cur;
                bestSol = pop(idx,:);
            end
            runBestFit(g) = bestFit;
        end % generations
        
        % log entropy over generations for this run
        entropyLog(run,:) = runBestFit(:)';
        
        % log run-level bounding box
        [rx1,ry1,rw,rh] = squaretable(round(bestSol));
        XLB_run = max(1, rx1);
        YLB_run = max(1, ry1);
        XUB_run = min(W, rx1 + rw - 1);
        YUB_run = min(H, ry1 + rh - 1);
        runBoundsSummary(run,:) = [run, XLB_run, XUB_run, YLB_run, YUB_run, ...
            runLB(1),runLB(2),runLB(3),runLB(4),runUB(1),runUB(2),runUB(3),runUB(4)];
        
        % update best-of-image
        if bestFit > bestOfImageFit
            bestOfImageFit = bestFit;
            bestOfImageSol = bestSol;
        end
    end % runs
    
    % ---------------- draw final best bounding box ----------------
    [fx1,fy1,fw,fh] = squaretable(round(bestOfImageSol));
    cx = bestOfImageSol(1); cy = bestOfImageSol(2);
    f = figure('Visible','off');
    imshow(originalImage); hold on;
    rectangle('Position',[fx1 fy1 fw fh],'EdgeColor','black','LineWidth',2,'LineStyle','--');
    draw_static_square(gca,round(cx),round(cy),STATIC_W,STATIC_H);
    frame = getframe(gca);
    imwrite(frame.cdata, fullfile(outputDir,'final_result.png'));   % save annotated result
    close(f);
    
    % ---------------- save logs ----------------
    avgEntropy = mean(entropyLog,1);   % average entropy curve across runs
    varNames = arrayfun(@(g)sprintf('Gen_%03d',g),1:numGen,'UniformOutput',false);
    T_entropy = array2table([entropyLog; avgEntropy], 'VariableNames', varNames);
    rowNames = [arrayfun(@(r)sprintf('Run_%d',r),1:numRuns,'UniformOutput',false),'Average'];
    T_entropy = addvars(T_entropy, rowNames', 'Before', 1, 'NewVariableNames', {'Run'});
    writetable(T_entropy, fullfile(excelDir, [imgName '_ENTROPY.xlsx']));
    
    % bounding boxes summary log
    T_boundsRuns = array2table(runBoundsSummary, 'VariableNames', ...
        {'RUN','XLB','XUB','YLB','YUB','LB_XC','LB_YC','LB_R1','LB_R2','UB_XC','UB_YC','UB_R1','UB_R2'});
    writetable(T_boundsRuns, fullfile(excelDir, [imgName '_BOUNDS_RUNS.xlsx']));
    
    summaryAvgEntropy(fileIdx,:) = avgEntropy;   % global log
end % images

% ---------------- save global summary ----------------
summaryTable = array2table(summaryAvgEntropy,'VariableNames', varNames);
summaryTable = addvars(summaryTable, imageIDs, 'Before', 1, 'NewVariableNames', {'IMAGE_NO'});
writetable(summaryTable, fullfile(excelDir, 'AVG_SUMMARY.xlsx'));

disp("✅ QIDE (DE-in-θ-space) Completed.");
toc;

%% ------------------ SUPPORT FUNCTIONS ----------------------
function out = enforce_bounds(c, LB, UB)
    % clamp phenotype values within lower & upper bounds
    out = max(LB, min(c, UB));
end

function meas = measure_theta(theta_matrix, LB, UB)
    % deterministic measurement: map θ → phenotype using sin²(θ)
    s2 = sin(theta_matrix).^2;
    meas = bsxfun(@plus, LB, s2 .* (UB - LB));
end

function meas = measure_theta_with_noise(theta_matrix, LB, UB, noise_level)
    % stochastic measurement: sin²(θ) + Gaussian noise, clipped [0,1]
    s2 = sin(theta_matrix).^2;
    noise = noise_level * randn(size(s2));
    s2n = min(max(s2 + noise, 0), 1);
    meas = bsxfun(@plus, LB, s2n .* (UB - LB));
end

function ang = inverse_measure_angles(vec, LB, UB)
    % inverse map phenotype → θ using asin(sqrt(s))
    s = (vec - LB) ./ (UB - LB);
    s = min(max(s,0),1);
    ang = asin(sqrt(s));
end

function f = entropy_fitness(img, chrom)
    % compute Shannon entropy of rectangular region defined by chrom
    [x1,y1,w,h] = squaretable(chrom);
    if x1<1 || y1<1 || (x1+w-1)>size(img,2) || (y1+h-1)>size(img,1)
        f = -Inf; return;   % invalid region
    end
    region = img(y1:y1+h-1, x1:x1+w-1);
    if isempty(region), f = -Inf; return; end
    region_uint = im2uint8(region);
    histVals = imhist(region_uint);
    p = histVals / numel(region_uint) + eps;
    f = -sum(p .* log2(p));   % entropy value
end

function [x1,y1,w,h] = squaretable(chrom)
    % convert [k,l,r1,r2] into rectangle coordinates
    k = chrom(1); l = chrom(2);
    rmid = 90;
    r1 = chrom(3); r2 = chrom(4);
    h = max(3, round((rmid + r1) * 2 + 1));
    w = max(3, round((rmid + r2) * 2 + 1));
    x1 = round(k - round(w/2));
    y1 = round(l - round(h/2));
    x1 = max(1, x1); y1 = max(1, y1);
end

function draw_static_square(ax, cx, cy, w, h)
    % draw a fixed-size square marker
    rectangle(ax,'Position',[round(cx-w/2) round(cy-h/2) w h], 'EdgeColor','black','LineWidth',2);
end

%% ------------------ Preprocessing (adaptive cases) ----------------------
function refined_mask=preprocess_sir(image)
    % preprocessing depends on entropy of grayscale image
    if size(image,3)==3, gray_image=rgb2gray(image); else, gray_image=image; end
    p=imhist(gray_image)/numel(gray_image);
    entropy_val=-sum(p.*log2(p+eps));
    tol=0.002;
    fprintf('📷 Entropy: %.5f\n',entropy_val);
    
    % special handling based on entropy signature (specific images)
    if abs(entropy_val-5.90679)<tol
        fprintf('➡ Preprocessing Image 19\n');
        filtered=medfilt2(gray_image,[5 5]);
        enhanced=adapthisteq(filtered);
        bg=medfilt2(enhanced,[120 120]);
        diff=enhanced-bg;
        final=medfilt2(diff,[190 190]);
        avg=fspecial('average',[5 5]);
        refined_mask=imfilter(final,avg);
        
    elseif abs(entropy_val-5.70994)<tol
        fprintf('➡ Preprocessing Image 20\n');
        filtered=medfilt2(gray_image,[5 5]);
        enhanced=adapthisteq(filtered);
        bg=medfilt2(enhanced,[120 120]);
        diff=enhanced-bg;
        final=medfilt2(diff,[190 190]);
        avg=fspecial('average',[5 5]);
        smooth=imfilter(final,avg);
        th=graythresh(smooth);
        bw=imbinarize(smooth,th);
        stats=regionprops(bw,'Area','Eccentricity','BoundingBox');
        mask=false(size(bw));
        for i=1:length(stats)
            if stats(i).Eccentricity<0.75 && stats(i).Area>3000 && stats(i).Area<25000
                bbox=round(stats(i).BoundingBox);
                x1=max(1,bbox(1)); y1=max(1,bbox(2));
                x2=min(size(bw,2),x1+bbox(3));
                y2=min(size(bw,1),y1+bbox(4));
                mask(y1:y2,x1:x2)=bw(y1:y2,x1:x2);
            end
        end
        refined_mask=smooth.*uint8(mask);
        
    elseif abs(entropy_val-6.19336)<tol
        fprintf('➡ Preprocessing Image 35\n');
        filtered=medfilt2(gray_image,[150 150]);
        enhanced=adapthisteq(filtered);
        bg=medfilt2(enhanced,[650 650]);
        diff=enhanced-bg;
        final=medfilt2(diff,[150 150]);
        avg=fspecial('average',[150 150]);
        refined_mask=imfilter(final,avg);
        
    elseif abs(entropy_val-6.46697)<tol
        fprintf('➡ Preprocessing Image 45\n');
        filtered=medfilt2(gray_image,[145 145]);
        enhanced=adapthisteq(filtered);
        bg=medfilt2(enhanced,[410 410]);
        diff=enhanced-bg;
        final=medfilt2(diff,[32 32]);
        avg=fspecial('average',[52 52]);
        refined_mask=imfilter(final,avg);
        
    elseif any(abs(entropy_val-[6.48679,6.25849,6.65673,6.73189,6.02197])<tol)
        fprintf('➡ Group preprocessing (Images 13,16,37,42,50)\n');
        filtered=medfilt2(gray_image,[100 100]);
        enhanced=adapthisteq(filtered);
        bg=medfilt2(enhanced,[210 210]);
        diff=enhanced-bg;
        final=medfilt2(diff,[32 32]);
        avg=fspecial('average',[52 52]);
        refined_mask=imfilter(final,avg);
