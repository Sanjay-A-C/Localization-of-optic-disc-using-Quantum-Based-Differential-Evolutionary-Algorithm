function avgGenEntropy = AVG_GEN_DE(summaryExcelPath, saveFolder)
% AVG_GEN_DE - Computes and saves average entropy across images for each generation
%
% INPUT:
%   summaryExcelPath - full path to AVG_SUMMARY.xlsx
%   saveFolder        - (optional) folder to save FINAL_AVG_VECTOR.xlsx
%
% OUTPUT:
%   avgGenEntropy     - 1x200 vector of average entropies (GEN1 to GEN200)

    if nargin < 1
        summaryExcelPath = "D:\Backup 12.12.24\Desktop\optic_disc_localization\EVAL_27AUG25\QQODE\XSLT\AVG_SUMMARY.xlsx";
    end
    if nargin < 2
        saveFolder = "D:\Backup 12.12.24\Desktop\optic_disc_localization\EVAL_27AUG25\QQODE\XSLT";
    end

    if ~isfile(summaryExcelPath)
        error("File not found: %s", summaryExcelPath);
    end

    % === Load data and compute mean across rows ===
    T = readtable(summaryExcelPath);
    genData = T{:, 2:end};  % Drop IMAGE_NO (first column)
    avgGenEntropy = mean(genData, 1);  % 1x200 vector

    % === Display
    fprintf("✅ Average entropy across all images at each generation:\n");
    disp(avgGenEntropy);

    % === Save to Excel (.xlsx) properly ===
    headers = arrayfun(@(g) sprintf('Gen_%03d', g), 1:size(genData,2), 'UniformOutput', false);
    outTable = array2table(avgGenEntropy, 'VariableNames', headers);

    % Define full path with filename
    saveFilePath = fullfile(saveFolder, 'FINAL_AVG_VECTOR.xlsx');
    writetable(outTable, saveFilePath, 'FileType', 'spreadsheet');  % ✅ Forces .xlsx format
    fprintf("✅ Saved final average entropy vector to: %s\n", saveFilePath);

    % === Optional: Plot
    figure;
    plot(1:length(avgGenEntropy), avgGenEntropy, '-o', 'LineWidth', 1.5);
    xlabel('Generation');
    ylabel('Average Entropy');
    title('Average Entropy Across All Images (Per Generation)');
    grid on;
end
