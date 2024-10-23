clc
clear
%close all

%算例参数
Xmin = -1;
Xmax =  1;
Ymin = -1;
Ymax =  1;
x0 = -0.35;
y0 = 0.0;
sigma = 0.005;
K = 1e-3;

% 使用示例
configFilename = 'config.txt';
% 从配置文件读取参数
parameters = readConfigFromFile(configFilename);
% 从结构体提取参数
T_max = parameters.T_max;
Nt = parameters.Nt;
N = parameters.N;
T_span = parameters.T_span;
K = parameters.K;


% 后续计算
dt = T_max/Nt;
h = (Xmax - Xmin) / N;
X = Xmin + 0.5*h : h : Xmax - 0.5*h;
Y = X;
[X1, Y1] = meshgrid(X,Y);
%T = ["05", "1", "2", "4"];
%T = ["20", "40", "60", "80"];
T = ["80"];

%% 设置colorbar
%text = 'MPL_rainbow';
%text = 'NCV_jaisnd';
%text = 'NCV_bright';
%colormap(nclCM(text))
%colormap(nclCM('NCV_jaisnd'));
colorbar('Position', [0.93 0.11 0.02 0.815]);
caxis([-0.1,0.8]);  

%% 速度场%
[X2, Y2] = meshgrid(X(1:8:end),Y(1:8:end));
Vx = -4*Y2;
Vy =  4*X2;
quiver(X2, Y2, Vx, Vy);

%设置刻度
xlabel('X','FontSize',20,'FontName','Times New Roman')
ylabel('Y','FontSize',20,'FontName','Times New Roman')
xlim([Xmin Xmax]);
ylim([Ymin Ymax]);
xticks(Xmin:1:Xmax);
yticks(Ymin:1:Ymax);
set(gcf,'color','w');
set(gca,'FontName','Times New Roman','FontSize',18)%,'LineWidth',1.2);
saveas(gcf, 'ex1_velocity', 'epsc');
saveas(gcf, 'ex1_velocity', 'fig');

% exact sol
C_ext = exp(-((X1 - x0).^2 + (Y1 - y0).^2) / sigma);
surf(X1,Y1,C_ext);
view(2);
shading interp
set(gcf,'color','w');
%grid on;
colorbar('Ticks',0:0.2:1);
caxis([-0.15,1.0]); 
set(gca,'GridLineStyle','--');
xticks(Xmin:0.5:Xmax);
yticks(Ymin:0.5:Ymax);
xlabel('X','FontSize',20,'FontName','Times New Roman')
ylabel('Y','FontSize',20,'FontName','Times New Roman')
set(gca,'FontName','Times New Roman','FontSize',18)%,'LineWidth',1.2);
saveas(gcf, 'ex1_C_initial', 'epsc');
saveas(gcf, 'ex1_C_initial', 'fig');

%%
for i = 1:length(T)
%i = length(T);
    itime = T_span*2*i;
    t = itime * dt;
    
    %% Exact sol
    X_star =  X1 * cos(4 * t) + Y1 * sin(4 * t);
    Y_star = -X1 * sin(4 * t) + Y1 * cos(4 * t);
    C_ext = sigma / (sigma + 4 * K * t) * exp(-((X_star - x0).^2 + (Y_star - y0).^2) / (sigma + 4 * K * t));

    text = sprintf('ex1_ext_%d_%d_T%s', Nt, N, T(i));
    
    %%------------- 等高线图 ------------
    % contour(X1,Y1,C_ext);
    % set(gcf,'color','w');
    % colormap(nclCM('NCV_bright'))
    % name = strcat(text,'_contour');
    % saveas(gcf, name, 'epsc');
    % saveas(gcf, name, 'fig');
    
    surf(X1, Y1, C_ext);
    shading interp
    % %------------- 3D 图 --------------
    % view(-116,54);
    % shading interp
    % set(gcf,'color','w');
    % colormap('default')
    % name = strcat(text,'_3D');
    % saveas(gcf, name, 'epsc');
    % saveas(gcf, name, 'fig');
    % 
    % %------------- 俯视图 --------------
    view(2);
    xlabel('x','FontSize',20,'FontName','Times New Roman')
    ylabel('y','FontSize',20,'FontName','Times New Roman')
    colorbar('Ticks',0:0.2:1);
    caxis([-0.15, 1.0]); 
    xticks(Xmin:0.5:Xmax);
    yticks(Ymin:0.5:Ymax);
    set(gca,'FontName','Times New Roman','FontSize',20)%,'LineWidth',1.2);
    name = strcat(text,'_view2');
    saveas(gcf, name, 'epsc');
    saveas(gcf, name, 'fig');

    %% T1S2
    % filename = sprintf('data_T1S2_%06d.txt',itime);
    % data = importdata(filename);
    % data1 = reshape(data(:,3),N,N);
    % text = sprintf('ex1_T1S2_CFV_%d_%d_T%s', Nt, N, T(i));
    
    %------------- 等高线图 ------------
    % contour(X1,Y1,data1);
    % set(gcf,'color','w');
    % name = strcat(text,'_contour');
    % saveas(gcf, name, 'epsc');
    % %saveas(gcf, name, 'fig');

    % surf(X1, Y1, data1);
    % shading interp
    % %------------- 3D 图 --------------
    % set(gcf,'color','w');
    % name = strcat(text,'_3D');
    % saveas(gcf, name, 'epsc');
    % saveas(gcf, name, 'fig');

    % %------------- 俯视图 --------------
    % view(2);
    % name = strcat(text,'_view2');
    % saveas(gcf, name, 'epsc');
    % saveas(gcf, name, 'fig');

    %------------ 误差图---------------
    % surf(X1,Y1,data1 - C_ext)
    % xlabel('x')
    % ylabel('y')
    % zlabel('Z')
    % shading interp
    % saveas(gcf, 'ex1_T1S2_err', 'fig');

    %% T2S4
    filename = sprintf('data_T2S4_%06d.txt',itime);
    data = importdata(filename);
    data2 = reshape(data(:,3),N,N);
    text = sprintf('ex1_T2S4_CFV_%d_%d_T%s', Nt, N, T(i));
    
    %------------- 等高线图 ------------
    % zLevels = -1:0.1:1;
    % contour(X1,Y1,data2,zLevels);
    % set(gcf,'color','w');
    % %colormap(nclCM('NCV_bright'))
    % % if mod(i,2) == 0
    % %     colorbar
    % % end
    % xlabel('x')
    % ylabel('y')
    % name = strcat(text,'_contour');
    % saveas(gcf, name, 'epsc');
    % saveas(gcf, name, 'fig');
    
    surf(X1, Y1, data2);
    shading interp
    %------------- 3D 图 --------------
    % %view(-116,54);
    % view(-117,58);
    % xlabel('x')
    % ylabel('y')
    % set(gcf,'color','w');
    % %colormap('default');
    % set(gca,'GridLineStyle','--');
    % colorbar
    % name = strcat(text,'_3D');
    % saveas(gcf, name, 'epsc');
    % saveas(gcf, name, 'fig');
    
    % 
    % %------------- 俯视图 --------------
    view(2);
    % if mod(i,2) == 0
    %     colorbar('Ticks',0:0.2:1);
    % end
    colorbar('Ticks',0:0.2:1);
    caxis([-0.15,1.0]); 
    xticks(Xmin:0.5:Xmax);
    yticks(Ymin:0.5:Ymax);
    xlabel('x','FontSize',20,'FontName','Times New Roman')
    ylabel('y','FontSize',20,'FontName','Times New Roman')
    set(gca,'FontName','Times New Roman','FontSize',20)%,'LineWidth',1.2);
    name = strcat(text,'_view2');
    saveas(gcf, name, 'epsc');
    saveas(gcf, name, 'fig');

    %------------ 误差图---------------
    ax1 = subplot(1,2,1);
    surf(X1,Y1,data2 - C_ext);
    axis equal;
    xlabel('x','FontSize',12,'FontName','Times New Roman')
    ylabel('y','FontSize',12,'FontName','Times New Roman')
    xticks(Xmin:0.5:Xmax);
    yticks(Ymin:0.5:Ymax);
    shading interp
    view(2);
    %colorbar;
    set(gca,'FontName','Times New Roman','FontSize',12)%,'LineWidth',1.2);
    %name = strcat(text,'_err');
    %saveas(gcf, name, 'epsc');
    %saveas(gcf, name, 'fig');

    %------------ 局部放大误差图---------------
    ax2 = subplot(1,2,2);
    surf(X1,Y1,data2 - C_ext);
    axis equal;
    axis([-0.2 0.4 -0.6 0.0]); % 根据需要设置放大的区域
    xlabel('x','FontSize',12,'FontName','Times New Roman')
    ylabel('y','FontSize',12,'FontName','Times New Roman')
    xticks(-0.2:0.2:0.4);
    yticks(-0.6:0.2:0.0);
    shading interp
    view(2);
    colorbar;
    set(gca,'FontName','Times New Roman','FontSize',12)%,'LineWidth',1.2);

    % 获取子图的当前位置
    pos1 = get(ax1, 'Position');
    set(ax2, 'Position', [pos1(1) + 0.1 + pos1(3), pos1(2), pos1(3), pos1(4)]); % 修改X坐标

    % 调整图形窗口大小
    set(gcf, 'Position', [100, 100, 1000, 400]); % 增加图形窗口的宽度

    name = strcat(text,'_err_zoom');
    saveas(gcf, name, 'epsc');
    saveas(gcf, name, 'fig');
    
    %% ------------ 切片图---------------
    % X0_row = N/2;
    % slice_data = C_ext(:,X0_row); % Extract the data at y == 0
    % plot(Y, slice_data, "Color","#EDB120",...
    % 'LineWidth', 1.4); 
    % 
    % hold on
    % 
    % slice_data1 = data1(:,X0_row); % Extract the data at y == 0
    % plot(Y, slice_data1, "--*", 'color', "#4DBEEE", ...
    %     'MarkerSize', 6, 'LineWidth', 0.8, "MarkerIndices",1:2:length(Y));
    % 
    % hold on
    % 
    % slice_data2 = data2(:,X0_row); % Extract the data at y == 0
    % plot(Y, slice_data2, ':.r', ... %'MarkerFaceColor', 'r' ...
    %      'LineWidth', 1.1, 'MarkerSize', 14, "MarkerIndices", 1:2:length(Y));
    % %scatter(Y(1:2:end), slice_data2(1:2:end),'filled',"red",'SizeData', 22);
    % 
    % hold on %被遮住再画一遍
    % %plot(Y, slice_data1, "b*", 'MarkerSize', 6, "MarkerIndices",1:2:length(Y));
    % 
    % %zticks[]
    % daspect([2,1,1]);
    % hLegend = legend({'exact sol','T1S2-CFV-Splt','T2S4-CC-CFV-Split'},'Location','northeast');
    % set(hLegend, 'FontSize', 8); % Adjust the font size as needed
    % name = sprintf('ex1_T%s_slice', T(i));
    % saveas(gcf, name, 'epsc');
    % saveas(gcf, name, 'fig');
    % 
    % hold off

    
end


