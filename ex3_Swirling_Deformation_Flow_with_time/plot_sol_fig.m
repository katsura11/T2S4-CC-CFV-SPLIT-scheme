clc
clear
close all

% 使用示例
configFilename = 'config.txt';
% 从配置文件读取参数
parameters = readConfigFromFile(configFilename);
% 从结构体提取参数
T_max = parameters.T_max;
Nt = parameters.Nt;
N = parameters.N;
T_span = parameters.T_span;

Xmin = -pi;
Xmax =  pi;
Ymin = -pi;
Ymax =  pi;
x0 = 0.3*pi;
y0 = 0.0;

% 后续计算
dt = T_max/Nt;
h = (Xmax - Xmin) / N;
X = Xmin + 0.5*h : h : Xmax - 0.5*h;
Y = X;
[X1, Y1] = meshgrid(X,Y);
%T = ["05", "1", "2", "4"];
T = ["05", "1", "2", "4"];
sigma = 0.385;

%% 设置colorbar
%text = 'MPL_rainbow';
%text = 'NCV_jaisnd';
%text = 'NCV_bright';
%colormap(nclCM(text))
%colormap(nclCM('NCV_jaisnd'));
colorbar('Position', [0.93 0.11 0.02 0.815]);


%% 速度场%
[X2, Y2] = meshgrid(X(1:16:end),Y(1:16:end));
r = sqrt(X2.^2+Y2.^2);
Vx = -2*pi*cos(X2/2).^2.*sin(Y2);
Vy =  2*pi*cos(Y2/2).^2.*sin(X2);
quiver(X2, Y2, Vx, Vy);
%streamslice(X2, Y2, Vx, Vy);

%设置刻度
xlabel('x','FontSize',20,'FontName','Times New Roman')
ylabel('y','FontSize',20,'FontName','Times New Roman')
xlim([Xmin Xmax]);
ylim([Ymin Ymax]);
%xticks(Xmin:1:Xmax);
%yticks(Ymin:1:Ymax);
set(gcf,'color','w');
saveas(gcf, 'ex3_velocity', 'epsc');
saveas(gcf, 'ex3_velocity', 'fig');

%% exact sol
filename = sprintf('data_T2S4_%06d.txt',0);
data = importdata(filename);
C_initial = reshape(data(:,3),N,N);
surf(X1,Y1,C_initial);
%view(-116,38);
shading interp
set(gcf,'color','w');
%grid on;
xlim([Xmin Xmax]);
ylim([Ymin Ymax]);
xlabel('x','FontSize',20,'FontName','Times New Roman')
ylabel('y','FontSize',20,'FontName','Times New Roman')
set(gca,'GridLineStyle','--');
saveas(gcf, 'ex3_C_initial', 'epsc');
saveas(gcf, 'ex3_C_initial', 'fig');

max_t1s2 = zeros(length(T),1);
min_t1s2 = zeros(length(T),1);
max_t2s4 = zeros(length(T),1);
min_t2s4 = zeros(length(T),1);


%%
for i = 1:length(T)
%i = length(T);
    itime = T_span*2^(i-1);
    t = itime * dt;

    %% T1S2_CFV
    filename = sprintf('data_T1S2_%06d.txt',itime);
    data = importdata(filename);
    data1 = reshape(data(:,3),N,N);
    text = sprintf('ex3_T1S2_CFV_%d_%d_T%s', Nt, N, T(i));

    max_t1s2(i) = max(max(data1));
    min_t1s2(i) = min(min(data1));
    
    %%------------- 等高线图 ------------
    % contour(X1,Y1,data1);
    % set(gcf,'color','w');
    % name = strcat(text,'_contour');
    % saveas(gcf, name, 'epsc');
    %saveas(gcf, name, 'fig');

    %------------- 3D 图 --------------
    surf(X1, Y1, data1);
    % shading interp
    % view(-46,50);
    % xlabel('x','FontSize',20,'FontName','Times New Roman')
    % ylabel('y','FontSize',20,'FontName','Times New Roman')
    % set(gcf,'color','w');
    % %colormap('default');
    % set(gca,'GridLineStyle','--');
    % if mod(i,2) == 0
    %     colorbar
    % end
    % xlim([Xmin Xmax]);
    % ylim([Ymin Ymax]);
    % name = strcat(text,'_3D');
    % saveas(gcf, name, 'epsc');
    % saveas(gcf, name, 'fig');

    %------------- 俯视图 --------------
    view(2);
    if i == length(T)
       colorbar
    end
    caxis([-0.1,1]); 
    xlim([Xmin Xmax]);
    ylim([Ymin Ymax]);
    xticks(Ymin:pi/2:Ymax);
    xticklabels({'-\pi','-\pi/2','0','\pi/2','\pi'})
    yticks(Ymin:pi/2:Ymax);
    yticklabels({'-\pi','-\pi/2','0','\pi/2','\pi'})
    xlabel('x','FontSize',20,'FontName','Times New Roman')
    ylabel('y','FontSize',20,'FontName','Times New Roman')
    set(gca,'FontName','Times New Roman','FontSize',20)%,'LineWidth',1.2);
    name = strcat(text,'_view2');
    saveas(gcf, name, 'epsc');
    saveas(gcf, name, 'fig');

    %% T1S2_CFV_DD
    % filename = sprintf('data_T1S2_DD_%06d.txt',itime);
    % data = importdata(filename);
    % data2 = reshape(data(:,3),N,N);
    % text = sprintf('ex3_T1S2_CFV_DD_%d_%d_T%s', Nt, N, T(i));
    
    %%------------- 等高线图 ------------
    % contour(X1,Y1,data1);
    % set(gcf,'color','w');
    % name = strcat(text,'_contour');
    % saveas(gcf, name, 'epsc');
    %saveas(gcf, name, 'fig');

    %%------------- 3D 图 --------------
    % surf(X1, Y1, data2);
    % shading interp
    % view(40,50);
    % xlabel('x','FontSize',20,'FontName','Times New Roman')
    % ylabel('y','FontSize',20,'FontName','Times New Roman')
    % set(gcf,'color','w');
    % colormap('default');
    % set(gca,'GridLineStyle','--');
    % if mod(i,2) == 0
    %     colorbar
    % end
    % xlim([Xmin Xmax]);
    % ylim([Ymin Ymax]);
    % name = strcat(text,'_3D');
    % saveas(gcf, name, 'epsc');
    % saveas(gcf, name, 'fig');

    % %------------- 俯视图 --------------
    % view(2);
    % %if mod(i,2) == 0
    % if i == length(T)
    %     colorbar
    % end
    % caxis([-0.1,1]); 
    % xlim([Xmin Xmax]);
    % ylim([Ymin Ymax]);
    % xticks(Ymin:pi/2:Ymax);
    % xticklabels({'-\pi','-\pi/2','0','\pi/2','\pi'})
    % yticks(Ymin:pi/2:Ymax);
    % yticklabels({'-\pi','-\pi/2','0','\pi/2','\pi'})
    % %xlabel('x','FontSize',20,'FontName','Times New Roman')
    % %ylabel('y','FontSize',20,'FontName','Times New Roman')
    % set(gca,'FontName','Times New Roman','FontSize',30)%,'LineWidth',1.2);
    % name = strcat(text,'_view2');
    % saveas(gcf, name, 'epsc');
    %saveas(gcf, name, 'fig');

    %% T2S4
    filename = sprintf('data_T2S4_%06d.txt',itime);
    data = importdata(filename);
    data3 = reshape(data(:,3),N,N);
    text = sprintf('ex3_T2S4_CFV_%d_%d_T%s', Nt, N, T(i));

    max_t2s4(i) = max(max(data3));
    min_t2s4(i) = min(min(data3));
    
    %------------- 等高线图 ------------
    %zLevels = -1:0.1:1;
    %contour(X1,Y1,data2,zLevels);
    %contour(X1,Y1,data2);
    %set(gcf,'color','w');
    %colormap(nclCM('NCV_bright'))
    % if mod(i,2) == 0
    %     colorbar
    % end
    %xlabel('x')
    %ylabel('y')
    %name = strcat(text,'_contour');
    %saveas(gcf, name, 'epsc');
    %saveas(gcf, name, 'fig');

    %------------- 3D 图 --------------
    surf(X1, Y1, data3);
    % shading interp
    % % %view(-116,54);
    % %view(-117,58);
    % view(-43,28);
    % xlabel('x','FontSize',20,'FontName','Times New Roman')
    % ylabel('y','FontSize',20,'FontName','Times New Roman')
    % set(gcf,'color','w');
    % %colormap('default');
    % set(gca,'GridLineStyle','--');
    % if mod(i,2) == 0
    %     colorbar
    % end
    % xlim([Xmin Xmax]);
    % ylim([Ymin Ymax]);
    % name = strcat(text,'_3D');
    % saveas(gcf, name, 'epsc');
    % saveas(gcf, name, 'fig');
    
    % 
    % %------------- 俯视图 --------------
    view(2);
    %if mod(i,2) == 0
    if i == length(T)
        colorbar
    end
    caxis([-0.1,1]); 
    xlim([Xmin Xmax]);
    ylim([Ymin Ymax]);
    xticks(Ymin:pi/2:Ymax);
    xticklabels({'-\pi','-\pi/2','0','\pi/2','\pi'})
    yticks(Ymin:pi/2:Ymax);
    yticklabels({'-\pi','-\pi/2','0','\pi/2','\pi'})
    xlabel('x','FontSize',20,'FontName','Times New Roman')
    ylabel('y','FontSize',20,'FontName','Times New Roman')
    set(gca,'FontName','Times New Roman','FontSize',20)%,'LineWidth',1.2);
    name = strcat(text,'_view2');
    saveas(gcf, name, 'epsc');
    saveas(gcf, name, 'fig');
    %% ------------ 切片图---------------
    % X0_row = N/2;
    % slice_data = C_initial(:,X0_row); % Extract the data at y == 0
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
    % name = sprintf('ex3_T%s_slice', T(i));
    % saveas(gcf, name, 'epsc');
    % saveas(gcf, name, 'fig');
    % 
    % hold off

    
end

print("max_t1s2 = ", max_t1s2);


