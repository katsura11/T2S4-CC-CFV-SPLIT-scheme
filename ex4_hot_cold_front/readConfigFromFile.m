function params = readConfigFromFile(filename)
    % 打开文件
    fileID = fopen(filename, 'r');
    
    % 初始化参数结构
    params = struct();
    
    % 读取文件，直到文件结束
    while ~feof(fileID)
        line = fgetl(fileID);
        if ~ischar(line) || isempty(line) || line(1) == '%' || strcmpi(line, 'over')
            % 跳过空行、注释行和结束行
            continue;
        end

        % 分割键和值
        [key, value] = strtok(line, ':');
        value = strtrim(strrep(value, ':', ''));

        % 将值转换为数值类型
        if ~isempty(value)
            value = str2double(value);
            if isnan(value)
                error('Invalid value in config file');
            end
        else
            error('Missing value for key in config file');
        end

        % 存储到结构体中
        params.(strtrim(key)) = value;
    end
    
    % 关闭文件
    fclose(fileID);
end
    



