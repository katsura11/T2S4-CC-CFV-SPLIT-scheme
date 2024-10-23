%ex23clc
clear
%close all

% 使用示例
configFilename = 'config.txt';
% 从配置文件读取参数
parameters = readConfigFromFile(configFilename);
% 从结构体提取参数
T_max = parameters.T_max;
Nt = parameters.Nt;
N = parameters.N;
T_span = parameters.T_span;
Xmin = -4;
Xmax =  4;
Ymin = -4;
Ymax =  4;

% 后续计算
dt = T_max/Nt;
h = (Xmax - Xmin) / N;
X = Xmin + 0.5*h : h : Xmax - 0.5*h;
Y = X;
[X1, Y1] = meshgrid(X,Y);
%T = ["05", "1", "2", "4"];
T = ["1", "2", "4", "8"];
sigma = 0.385;

%% 设置colorbar
%text = 'MPL_rainbow';
%text = 'NCV_jaisnd';
%text = 'NCV_bright';
%colormap(nclCM(text))
colorbar('Position', [0.93 0.11 0.02 0.815]);

%% 速度场%
[X2, Y2] = meshgrid(X(1:4:end),Y(1:4:end));
r = sqrt(X2.^2+Y2.^2);
f_t = tanh(r)./cosh(r).^2;
Vx = -Y2./r.*f_t/sigma;
Vy =  X2./r.*f_t/sigma;
quiver(X2, Y2, Vx, Vy);

%设置刻度
xlabel('x','FontSize',12,'FontName','Times New Roman')
ylabel('y','FontSize',12,'FontName','Times New Roman')
set(gca,'FontName','Times New Roman','FontSize',20)
xlim([Xmin Xmax]);
ylim([Ymin Ymax]);
xticks(Xmin:2:Xmax);
yticks(Ymin:2:Ymax);
set(gcf,'color','w');
saveas(gcf, 'ex4_velocity', 'epsc');
saveas(gcf, 'ex4_velocity', 'fig');

%% exact sol
C_ext = -tanh(Y1/2);
surf(X1,Y1,C_ext);
view(-117,58);
shading interp
xlim([Xmin Xmax]);
ylim([Ymin Ymax]);
xticks(Xmin:2:Xmax);
yticks(Ymin:2:Ymax);
xlabel('x','FontSize',10,'FontName','Times New Roman')
ylabel('y','FontSize',10,'FontName','Times New Roman')
set(gca,'FontName','Times New Roman','FontSize',22)
set(gcf,'color','w');
colorbar('Ticks',-1:0.4:1);
%grid on;
set(gca,'GridLineStyle','--');
saveas(gcf, 'ex4_C_initial', 'epsc');
saveas(gcf, 'ex4_C_initial', 'fig');


%%
for i = 1:length(T)
%i = length(T);
    itime = T_span*2^(i-1);
    t = itime * dt;

    %% Exact sol
    C_ext = cal_exact_sol(N, Nt, t);
    %C_ext = -tanh(Y1/2.*cos(f_t*dt*t./(0.385*r)) - X1/2.*sin(f_t*dt*t./(0.385*r)));
    
    %------------- 等高线图 ------------
    % contour(X1,Y1,C_ext);
    % set(gcf,'color','w');
    % colormap(nclCM('NCV_bright'))
    % text = sprintf('ex4_ext_%d_%d_T%s', Nt, N, T(i));
    % name = strcat(text,'_contour');
    % saveas(gcf, name, 'epsc');
    % saveas(gcf, name, 'fig');
    % 
    % %------------- 3D 图 --------------
    % surf(X1, Y1, C_ext);
    % view(-116,54);
    % shading interp
    % set(gcf,'color','w');
    % colormap('default')
    % name = strcat(text,'_3D');
    % saveas(gcf, name, 'epsc');
    % saveas(gcf, name, 'fig');
    % 
    % %------------- 俯视图 --------------
    % view(2);
    % name = strcat(text,'_view2');
    % saveas(gcf, name, 'epsc');
    %saveas(gcf, name, 'fig');

    %% T1S2
    filename = sprintf('data_T1S2_%06d.txt',itime);
    data = importdata(filename);
    data1 = reshape(data(:,3),N,N);
    text = sprintf('ex4_T1S2_CFV_%d_%d_T%s', Nt, N, T(i));
    
    %------------- 等高线图 ------------
    % zLevels = -1:0.1:1;
    % contour(X1,Y1,data1,zLevels);
    % %if mod(i,2) == 0
    % if i == length(T)
    %     colorbar('Ticks',-1:0.4:1);
    % end
    % set(gcf,'color','w');
    % %xlim([Xmin Xmax]);
    % %ylim([Ymin Ymax]);
    % xticks(Xmin:2:Xmax);
    % yticks(Ymin:2:Ymax);
    % %xlabel('x','FontSize',20,'FontName','Times New Roman')
    % %ylabel('y','FontSize',20,'FontName','Times New Roman')
    % set(gca,'FontName','Times New Roman','FontSize',22)%,'LineWidth',1.2);
    % name = strcat(text,'_contour');
    % saveas(gcf, name, 'epsc');
    % saveas(gcf, name, 'fig');
    % 
    % %------------- 3D 图 --------------
    % surf(X1, Y1, data1);
    % view(-117,58);
    % shading interp
    % if i == length(T)
    %     colorbar('Ticks',-1:0.4:1);
    % end
    % set(gcf,'color','w');
    % xlim([Xmin Xmax]);
    % ylim([Ymin Ymax]);
    % xticks(Xmin:2:Xmax);
    % yticks(Ymin:2:Ymax);
    % xlabel('x','FontSize',10,'FontName','Times New Roman')
    % ylabel('y','FontSize',10,'FontName','Times New Roman')
    % set(gca,'FontName','Times New Roman','FontSize',22)%,'LineWidth',1.2);
    % name = strcat(text,'_3D');
    %saveas(gcf, name, 'epsc');
    %saveas(gcf, name, 'fig');

    % %------------- 俯视图 --------------
    % view(2);
    % name = strcat(text,'_view2');
    %saveas(gcf, name, 'epsc');
    %saveas(gcf, name, 'fig');

    %------------ 误差图---------------
    % surf(X1,Y1,data1 - C_ext)
    % xlabel('X')
    % ylabel('Y')
    % zlabel('Z')
    % shading interp
    % saveas(gcf, 'ex4_T1S2_err', 'fig');

    %% T1S2_DD
    % filename = sprintf(['data_T1S2_DD_' ...
    %     '%06d.txt'],itime);
    % data = importdata(filename);
    % data2 = reshape(data(:,3),N,N);
    % text = sprintf('ex4_T1S2_CFV_DD_%d_%d_T%s', Nt, N, T(i));
    % 
    % %------------- 等高线图 ------------
    % contour(X1,Y1,data2);
    % if mod(i,2) == 0
    %     colorbar
    % end
    % set(gcf,'color','w');
    % name = strcat(text,'_contour');
    % saveas(gcf, name, 'epsc');
    % saveas(gcf, name, 'fig');
    % % 
    % %------------- 3D 图 --------------
    % surf(X1, Y1, data2);
    % shading interp
    % set(gcf,'color','w');
    % name = strcat(text,'_3D');
    % saveas(gcf, name, 'epsc');
    % saveas(gcf, name, 'fig');
    % 
    % % %------------- 俯视图 --------------
    % view(2);
    % name = strcat(text,'_view2');
    % saveas(gcf, name, 'epsc');
    % saveas(gcf, name, 'fig');

    %------------ 误差图---------------
    % surf(X1,Y1,data2 - C_ext)
    % xlabel('X')
    % ylabel('Y')
    % zlabel('Z')
    % shading interp
    % saveas(gcf, 'ex4_T1S2_err', 'fig');

    %% T2S4
    filename = sprintf('data_T2S4_%06d.txt',itime);
    data = importdata(filename);
    data3 = reshape(data(:,3),N,N);
    
    %------------- 等高线图 ------------
    % zLevels = -1:0.1:1;
    % contour(X1,Y1,data3,zLevels);
    % set(gcf,'color','w');
    % %colormap(nclCM('NCV_bright'))
    % %if mod(i,2) == 0
    % if i == length(T)
    %     colorbar('Ticks',-1:0.4:1);
    % end
    % %xlim([Xmin Xmax]);
    % %ylim([Ymin Ymax]);
    % xticks(Xmin:2:Xmax);
    % yticks(Ymin:2:Ymax);
    % %xlabel('x','FontSize',20,'FontName','Times New Roman')
    % %ylabel('y','FontSize',20,'FontName','Times New Roman')
    % set(gca,'FontName','Times New Roman','FontSize',22)%,'LineWidth',1.2);
    % text = sprintf('ex4_T2S4_CFV_%d_%d_T%s', Nt, N, T(i));
    % name = strcat(text,'_contour');
    % saveas(gcf, name, 'epsc');
    % saveas(gcf, name, 'fig');

    %------------- 3D 图 --------------
    surf(X1, Y1, data3);
    %view(-116,54);
    view(-117,58);
    xlim([Xmin Xmax]);
    ylim([Ymin Ymax]);
    xticks(Xmin:2:Xmax);
    yticks(Ymin:2:Ymax);
    xlabel('x','FontSize',10,'FontName','Times New Roman')
    ylabel('y','FontSize',10,'FontName','Times New Roman')
    set(gca,'FontName','Times New Roman','FontSize',22)%,'LineWidth',1.2);
    shading interp
    set(gcf,'color','w');
    %colormap('default');
    set(gca,'GridLineStyle','--');
    %if mod(i,2) == 0
    if i == length(T)
        colorbar('Ticks',-1:0.4:1);
    end
    name = strcat(text,'_3D');
    saveas(gcf, name, 'epsc');
    saveas(gcf, name, 'fig');


    % % %------------- 俯视图 --------------
    view(2);
    %if mod(i,2) == 0
    if i == length(T)
        colorbar('Ticks',-1:0.4:1);
    end
    name = strcat(text,'_view2');
    %saveas(gcf, name, 'epsc');
    %saveas(gcf, name, 'fig');

     %------------ 误差图---------------
    % surf(X1,Y1,data3 - C_ext)
    % xlabel('X')
    % ylabel('Y')
    % zlabel('Z')
    % shading interp
    % saveas(gcf, 'ex4_T2S4_err', 'fig');
    
    % ------------ 切片图---------------
    X0_row = N/2;
    slice_data = C_ext(:,X0_row); % Extract the data at y == 0
    plot(Y, slice_data, "Color","#EDB120",...
    'LineWidth', 1.4); 

    hold on

    slice_data1 = data1(:,X0_row); % Extract the data at y == 0
    plot(Y, slice_data1, "-+", 'color', "#4DBEEE", ...
        'MarkerSize', 6, 'LineWidth', 0.8, "MarkerIndices",1:2:length(Y));

    hold on

    % slice_data2 = data2(:,X0_row); % Extract the data at y == 0
    % plot(Y, slice_data2, "--o", 'color', "#7E2F8E", ...
    %     'MarkerSize', 4, 'LineWidth', 0.8, "MarkerIndices",1:2:length(Y));

    slice_data3 = data3(:,X0_row); % Extract the data at y == 0
    plot(Y, slice_data3, ':.r', ... %'MarkerFaceColor', 'r' ...
         'LineWidth', 1.1, 'MarkerSize', 14, "MarkerIndices", 1:2:length(Y));
    %scatter(Y(1:2:end), slice_data2(1:2:end),'filled',"red",'SizeData', 22);


    %zticks[]
    daspect([2,1,1]);
    %hLegend = legend({'exact sol','T1S2-C-CFV-Splt','T1S2-C-CFV-DD','T2S4-CC-CFV-Split'},'Location','northeast');
    hLegend = legend({'exact sol','T1S2-C-CFV-Splt','T2S4-CC-CFV-Split'},'Location','northeast');
    set(hLegend, 'FontSize', 8); % Adjust the font size as needed
    name = sprintf('ex4_T%s_slice', T(i));
    saveas(gcf, name, 'epsc');
    saveas(gcf, name, 'fig');

    hold off

    
end


