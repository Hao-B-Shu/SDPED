function norm_AP = calcNormalizedAP(eval_txt_path)
% calcNormalizedAP: 读取或计算归一化 AP
% 
% 逻辑:
%   1. 提取 eval_txt_path 所在的目录路径。
%   2. 直接检查该目录下是否存在 Normalized_AP.txt。
%   3. 若存在，直接读取并返回文件中的 AP 值。
%   4. 若不存在，计算除以 ΔR 后的归一化 AP，创建 Normalized_AP.txt 写入并返回。

if nargin < 1 || isempty(eval_txt_path)
    error('请提供 eval_bdry_thr.txt 的有效文件路径！');
end

assert(exist(eval_txt_path, 'file') == 2, '未找到输入文件: %s', eval_txt_path);

%% 1. 获取目标保存路径并直接检查 Normalized_AP.txt 是否存在
[output_dir, ~, ~] = fileparts(eval_txt_path);
save_file_path = fullfile(output_dir, 'Normalized_AP.txt');

% 直接检查 Normalized_AP.txt 文件是否存在
if exist(save_file_path, 'file') == 2
    norm_AP = load(save_file_path);
    fprintf('直接读取: %s (AP = %.6f)\n', save_file_path, norm_AP);
    return;
end

%% 2. 若不存在，读取 eval_bdry_thr.txt 数据 (4列: Thr, Recall, Precision, F-measure)
data = load(eval_txt_path);
R = data(:, 2); % 第 2 列为 Recall
P = data(:, 3); % 第 3 列为 Precision

%% 3. 按 Recall 升序排序并计算原始积分 AP
[R, idx] = sort(R(:));
P = P(idx);
raw_AP = trapz(R, P); 

%% 4. 计算 Recall 的有效跨度 ΔR = R_max - R_min 并进行归一化
R_min = min(R);
R_max = max(R);
delta_R = R_max - R_min;

if delta_R > 1e-6
    norm_AP = raw_AP / delta_R;
else
    norm_AP = mean(P); % 防止除以 0
end

norm_AP = min(norm_AP, 1.0); % 保证数值不超过 1.0

%% 5. 建立 Normalized_AP.txt 并写入重算后的 AP 值
fid = fopen(save_file_path, 'w');
if fid ~= -1
    fprintf(fid, '%.6f\n', norm_AP);
    fclose(fid);
    fprintf('>>> 建立并保存归一化 AP 到文件: %s\n', save_file_path);
else
    warning('无法创建或写入文件: %s', save_file_path);
end

%% 6. 控制台输出计算信息
%fprintf('------------------------------------\n');
%fprintf('评估数据路径 : %s\n', eval_txt_path);
%fprintf('Recall 跨度   : [%.6f, %.6f] (ΔR = %.6f)\n', R_min, R_max, delta_R);
%fprintf('原始 AP   : %.6f\n', raw_AP);
%fprintf('Normalized AP : %.6f\n', norm_AP);
%fprintf('------------------------------------\n');

end

%calcNormalizedAP('eval_bdry_thr.txt')