clc
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

Xmin = -1;
Xmax =  1;
Ymin = -1;
Ymax =  1;
x0 = -0.35;
y0 = 0.0;
sigma = 0.005;

K = 1e-3;

% 后续计算
dt = T_max/Nt;
h = (Xmax - Xmin) / N;
X = Xmin + 0.5*h : h : Xmax - 0.5*h;
Y = X;
[X1, Y1] = meshgrid(X,Y);
%T = ["05", "1", "2", "4"];
T = ["20", "40", "80"];
%T = 100;

%% 设置colorbar
%text = 'MPL_rainbow';
%text = 'NCV_jaisnd';
%text = 'NCV_bright';
%colormap(nclCM(text))
%colormap(nclCM('NCV_jaisnd'));
colorbar('Position', [0.93 0.11 0.02 0.815]);
%caxis([-0.1,0.8]);  

%% 速度场%
[X2, Y2] = meshgrid(X(1:4:end),Y(1:4:end));
Vx = -pi*cos(0.5*pi*X2).*sin(0.5*pi*Y2);
Vy =  pi*cos(0.5*pi*Y2).*sin(0.5*pi*X2);
quiver(X2, Y2, Vx, Vy);

%设置刻度
xlabel('x')
ylabel('y')
xlim([Xmin Xmax]);
ylim([Ymin Ymax]);
xticks(Xmin:1:Xmax);
yticks(Ymin:1:Ymax);
set(gcf,'color','w');
saveas(gcf, 'ex2_velocity', 'epsc');

%% exact sol
C_ext = exp(-((X1 - x0).^2 + (Y1 - y0).^2) / sigma);
surf(X1,Y1,C_ext);
shading interp
set(gcf,'color','w');
xlabel('x')
ylabel('y')
%grid on;
colorbar
set(gca,'GridLineStyle','--');
saveas(gcf, 'ex2_C_initial', 'epsc');
saveas(gcf, 'ex2_C_initial', 'fig');


%%
for i = 1:length(T)
    itime = T_span*2^i;
    t = itime * dt;
  
    %% Exact sol
    X_star =  X1 * cos(pi * t) + Y1 * sin(pi * t);
    Y_star = -X1 * sin(pi * t) + Y1 * cos(pi * t);
    C_ext = sigma / (sigma + 4 * K * t) * exp(-((X_star - x0).^2 + (Y_star - y0).^2) / (sigma + 4 * K * t));

    text = sprintf('ex2_ext_%d_%d_T%s', Nt, N, T(i));
    
    %%------------- 等高线图 ------------
    % contour(X1,Y1,C_ext);
    % set(gcf,'color','w');
    % colormap(nclCM('NCV_bright'))
    % name = strcat(text,'_contour');
    % saveas(gcf, name, 'epsc');
    % saveas(gcf, name, 'fig');
    
    surf(X1, Y1, C_ext);
    % shading interp
    % %------------- 3D 图 --------------
    % set(gcf,'color','w');
    % colormap('default')
    % name = strcat(text,'_3D');
    % saveas(gcf, name, 'epsc');
    % saveas(gcf, name, 'fig');
    % 
    % %------------- 俯视图 --------------
    % view(2);
    % if mod(i,2) == 0
    %     %colorbar('Ticks',0:0.2:1);
    % end2
    % caxis([-0.1,0.8]); 
    % xticks(Xmin:0.5:Xmax);
    % yticks(Ymin:0.5:Ymax);
    % %xlabel('x','FontSize',20,'FontName','Times New Roman')
    % %ylabel('y','FontSize',20,'FontName','Times New Roman')
    % set(gca,'FontName','Times New Roman','FontSize',24)%,'LineWidth',1.2);
    % name = strcat(text,'_view2');
    % saveas(gcf, name, 'epsc');
    % saveas(gcf, name, 'fig');
    % 
    % % %------------- x-z 平面图 --------------
    % view(0,0);
    % if mod(i,2) == 0
    %     colorbar('Ticks',0:0.2:1);
    % end
    % caxis([-0.1,0.8]); 
    % xticks(Xmin:0.5:Xmax);
    % yticks(Ymin:0.5:Ymax);
    % %xlabel('x','FontSize',20,'FontName','Times New Roman')
    % %ylabel('y','FontSize',20,'FontName','Times New Roman')
    % set(gca,'FontName','Times New Roman','FontSize',24)%,'LineWidth',1.2);
    % name = strcat(text,'_xz2D');
    % saveas(gcf, name, 'epsc');
    % saveas(gcf, name, 'fig');
    % 

    %% T1S2
    filename = sprintf('data_T1S2_%06d.txt',itime);
    data = importdata(filename);
    data1 = reshape(data(:,3),N,N);
    text = sprintf('ex2_T1S2_CFV_%d_%d_T%s', Nt, N, T(i));
    
    %------------- 等高线图 ------------
    % contour(X1,Y1,data1);
    % set(gcf,'color','w');
    % name = strcat(text,'_contour');
    % saveas(gcf, name, 'epsc');
    % %saveas(gcf, name, 'fig');
    % 

    surf(X1, Y1, data1);
    shading interp
    % %------------- 3D 图 --------------
    % set(gcf,'color','w');
    % name = strcat(text,'_3D');
    % saveas(gcf, name, 'epsc');
    % saveas(gcf, name, 'fig');

    %------------- 俯视图 --------------
    % view(2);
    % caxis([-0.1,0.8]); 
    % xticks(Xmin:0.5:Xmax);
    % yticks(Ymin:0.5:Ymax);
    % %xlabel('x','FontSize',20,'FontName','Times New Roman')
    % %ylabel('y','FontSize',20,'FontName','Times New Roman')
    % set(gca,'FontName','Times New Roman','FontSize',24)%,'LineWidth',1.2);
    % name = strcat(text,'_view2');
    % saveas(gcf, name, 'epsc');
    % saveas(gcf, name, 'fig');
    % 
    % % %------------- x-z 平面图 --------------
    % view(0,0);
    % if mod(i,2) == 0
    %     colorbar('Ticks',0:0.2:1);
    % end
    % caxis([-0.1,0.8]); 
    % xticks(Xmin:0.5:Xmax);
    % yticks(Ymin:0.5:Ymax);
    % %xlabel('x','FontSize',20,'FontName','Times New Roman')
    % %ylabel('y','FontSize',20,'FontName','Times New Roman')
    % set(gca,'FontName','Times New Roman','FontSize',24)%,'LineWidth',1.2);
    % name = strcat(text,'_xz2D');
    % saveas(gcf, name, 'epsc');
    % saveas(gcf, name, 'fig');

    %------------ 误差图---------------
    surf(X1,Y1,data1 - C_ext)
    shading interp
    %view(0,0);
    %view(-50,1);
    view(2);
    colorbar;
    % xticks(Xmin:1:Xmax);
    % yticks(Ymin:1:Ymax);
    xlabel('x','FontSize',20,'FontName','Times New Roman')
    ylabel('y','FontSize',20,'FontName','Times New Roman')
    set(gca,'FontSize',18)%,'LineWidth',1.2);
    name = strcat(text,'_err');
    saveas(gcf, name, 'fig');
    saveas(gcf, name, 'eps');

    %% T2S4
    filename = sprintf('data_T2S4_%06d.txt',itime);
    data = importdata(filename);
    data2 = reshape(data(:,3),N,N);
    text = sprintf('ex2_T2S4_CFV_%d_%d_T%s', Nt, N, T(i));
    
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
    
    % surf(X1, Y1, data2);
    % shading interp
    %------------- 3D 图 --------------
    % %view(-116,54);
    % view(-117,58);
    % xlabel('x')
    % ylabel('y')
    % set(gcf,'color','w');
    % %colormap('default');
    % set(gca,'GridLineStyle','--');
    % if mod(i,2) == 0
    %     colorbar
    % end
    % name = strcat(text,'_3D');
    % saveas(gcf, name, 'epsc');
    % saveas(gcf, name, 'fig');
    
    % 
    % %------------- 俯视图 --------------
    % view(2);
    % if mod(i,2) == 0
    %     %colorbar('Ticks',0:0.2:1);
    % end
    % caxis([-0.1,0.8]);
    % xticks(Xmin:0.5:Xmax);
    % yticks(Ymin:0.5:Ymax);
    % %xlabel('x','FontSize',20,'FontName','Times New Roman')
    % %ylabel('y','FontSize',20,'FontName','Times New Roman')
    % set(gca,'FontName','Times New Roman','FontSize',24)%,'LineWidth',1.2);
    % name = strcat(text,'_view2');
    % saveas(gcf, name, 'epsc');
    % saveas(gcf, name, 'fig');
    % 
    % % %------------- x-z 平面图 --------------
    % view(0,0);
    % if mod(i,2) == 0
    %     colorbar('Ticks',0:0.2:1);
    % end
    % caxis([-0.1,0.8]); 
    % xticks(Xmin:0.5:Xmax);
    % yticks(Ymin:0.5:Ymax);
    % %xlabel('x','FontSize',20,'FontName','Times New Roman')
    % %ylabel('y','FontSize',20,'FontName','Times New Roman')
    % set(gca,'FontName','Times New Roman','FontSize',24)%,'LineWidth',1.2);
    % name = strcat(text,'_xz2D');
    % saveas(gcf, name, 'epsc');
    % saveas(gcf, name, 'fig');

    %------------ 误差图---------------
    surf(X1,Y1,data2 - C_ext)
    shading interp
    %view(0,0);
    %view(-50,1)
    view(2);
    colorbar;
    caxis([-0.01,0.01]); 
    % xticks(Xmin:0.5:Xmax);
    % yticks(Ymin:0.5:Ymax);
    xlabel('x','FontSize',20,'FontName','Times New Roman')
    ylabel('y','FontSize',20,'FontName','Times New Roman')
    set(gca,'FontSize',18)%,'LineWidth',1.2);
    name = strcat(text,'_err');
    saveas(gcf, name, 'fig');
    saveas(gcf, name, 'eps');
    
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
    % name = sprintf('ex2_T%s_slice', T(i));
    % saveas(gcf, name, 'epsc');
    % saveas(gcf, name, 'fig');
    % 
    % hold off

    
end


