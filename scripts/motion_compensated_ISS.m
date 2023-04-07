%% Render map of Gen4 data using full window
addpath("matlibs")
DATASET = "20220121a_Salvador_2022-01-21_20~58~34_NADIR.h5"; %"0.125";
load("/media/sam/Samsung_T52/PhD/Code/orbital_localisation/data/mat/" + DATASET + ".mat")

RECORD                  = 0;
PLOT                    = 1;
FTW                     = 10000;
STR                     = 10;
INTERPOLATION           = 1;
LINEAR                  = 0;

sigma                   = 1.0;
WINDOW                  = 4e5;
displayFreq             = 1e5;
tau                     = 2e5;
timeMax                 = 20e6;

ii = find(events(:,1)> 1e6 & events(:,1)< timeMax);
e  = struct("x",double(events(ii,2)),"y",double(events(ii,3)),"p",double(events(ii,4)),"ts",double(events(ii,1)));

speedx = 21.6;
speedy = -0.5;
vx                      = repmat(speedx,[numel(e.x),1]);
vy                      = repmat(speedy,[numel(e.x),1]);

if RECORD
    writerObj = VideoWriter("/media/sam/Samsung_T52/PhD/Dataset/NORALPH_ICNS_EB_Space_Imaging_Speed_Dataset/videos/" +DATASET + "_FTW_" + num2str(FTW) + "_STR_" + num2str(STR) +".avi");
    writerObj.FrameRate = 10;
    open(writerObj);
end

xMax    = max(e.x);
yMax    = max(e.y);
nEvents = numel(e.x);
S       = zeros(xMax,yMax); T = S; P = T;
Smc     = zeros(xMax+5,yMax+5); Tmc = Smc; Pmc = Tmc;
randV   = rand(1);

nextTimeSample  = e.ts(1,1)+displayFreq;

warpedx = round(e.x-vx(:,1).*e.ts/1e6);
warpedy = round(e.y-vy(:,1).*e.ts/1e6);

padding = double(ceil(1.0 * sigma));
kernel_indices = zeros(padding * 2 + 1, padding * 2 + 1, 2);

for y = 1:padding * 2 + 1
    for x = 1:padding * 2 + 1
        kernel_indices(y, x, 1) = x - padding;
        kernel_indices(y, x, 2) = y - padding;
    end
end

x_minimum = min(warpedx);
y_minimum = min(warpedy);

xs = warpedx - x_minimum + padding;
ys = warpedy - y_minimum + padding;

pixels = zeros(ceil(max(ys)) + padding + 1,ceil(max(xs)) + padding + 1);

xis = round(xs);
yis = round(ys);

xfs = xs - xis;
yfs = ys - yis;

sigma_factor = -1.0 / (2.0 * sigma^2.0);

fp          = 0;
for i = 1:numel(xis)
    x       = e.x(i)+1;
    y       = e.y(i)+1;
    t       = e.ts(i);
    p       = e.p(i);
    fieldx  = vx(i);
    fieldy  = vy(i);
    T(x,y)  = t;
    P(x,y)  = p;
    
    sumF = zeros(size(kernel_indices,1),size(kernel_indices,1),size(kernel_indices,3));
    for l = 1:size(kernel_indices,1)
        for j = 1:size(kernel_indices,1)
            summation = 0;
            for axisX  = 1:size(kernel_indices,3)
                summation = summation + kernel_indices(l,j,axisX);
                sumF(l,j,axisX) = summation;
            end
        end
    end
    sumF = sumF.^ 2.0;
    
    finalSummation = (sumF(:,:,1) + sumF(:,:,2))*sigma_factor;
    pixels(yis(i)-padding+1:yis(i)+padding+1,xis(i)-padding+1:xis(i)+padding+1) = pixels(yis(i)-padding+1:yis(i)+padding+1,xis(i)-padding+1:xis(i)+padding+1)+...
        exp(finalSummation);
    
    if t > nextTimeSample
        fp=fp+1;
        nextTimeSample = nextTimeSample + displayFreq;
        
        if LINEAR
            S = P.*(1+((T-t)/(2*tau)));
        else
            S = P.*exp((T-t)/tau);
        end
        
        deltat = e.ts(e.ts>nextTimeSample-displayFreq & e.ts<nextTimeSample) - nextTimeSample;
        xnew = e.x(e.ts>nextTimeSample-displayFreq & e.ts<nextTimeSample);
        ynew = e.y(e.ts>nextTimeSample-displayFreq & e.ts<nextTimeSample);
        deltaVX = vx(e.ts>nextTimeSample-displayFreq & e.ts<nextTimeSample,1);
        deltaVY = vy(e.ts>nextTimeSample-displayFreq & e.ts<nextTimeSample,1);
        
        warpednewx = round(xnew-deltaVX.*deltat/1e6);
        warpednewy = round(ynew-deltaVY.*deltat/1e6);
        
        x_minimum_warped  = min(warpednewx);
        y_minimum_warped  = min(warpednewy);
        
        xsw = warpednewx - x_minimum_warped + padding;
        ysw = warpednewy - y_minimum_warped + padding;
        
        %%%%%%%%%%%% ground speed
        x_minimum_warped_g  = min(xnew);
        y_minimum_warped_g  = min(ynew);
        
        xsw_g = xnew - x_minimum_warped_g + padding;
        ysw_g = ynew - y_minimum_warped_g + padding;
        
        motion_compensated_frame = accumulate(xsw,ysw);
        
        if PLOT
            figure(794546);
%             subtightplot(2,3,[1 4])
            imagesc(pixels.^(1/4));colormap(magma(100));axis off;hold on
%             %         text(10,60,"$Time \ Surface$",'Color', '#ffffff','interpreter','latex', 'FontWeight','bold','FontSize',20);
%             %         text(550,60,"$Unstable$",'Color', '#ffffff','interpreter','latex', 'FontWeight','bold','FontSize',20);
%             %         text(10,130,"$\tau: \ $"+num2str(tau/1e6)+"$s$",'Color', '#ffffff','interpreter','latex', 'FontWeight','bold','FontSize',20);
%             text(10,60,"$Accumulated \ Image \ H(u_i,\theta)$",'Color', '#ffffff','interpreter','latex', 'FontWeight','bold','FontSize',20);
%             text(550,60,"$Unstable$",'Color', '#FF0000','interpreter','latex', 'FontWeight','bold','FontSize',20);
%             text(10,130,"$\Delta t: \ $"+num2str(displayFreq/1e3)+"$ms$",'Color', '#ffffff','interpreter','latex', 'FontWeight','bold','FontSize',20);
%             text(10,200,"$t: \ $"+num2str(t/1e6)+"$s$",'Color', '#ffffff','interpreter','latex', 'FontWeight','bold','FontSize',20);
%             text(10,1250,DATASET,'FontSize',10,'Color', '#ffffff','interpreter','latex', 'FontWeight','bold');
%             subtightplot(2,3,[2 5]);
%             imagesc(imrotate(flip(motion_compensated_frame.^(1/1)),0));colormap(magma(100));axis off;hold on
%             text(10,60,"$Accumulated \ Image \ H(u_i,\theta)$",'Color', '#ffffff','interpreter','latex', 'FontWeight','bold','FontSize',20);
%             text(550,60,"$Stabilised$",'Color', '#FF0000','interpreter','latex', 'FontWeight','bold','FontSize',20);
%             text(10,130,"$\Delta t: \ $"+num2str(displayFreq/1e3)+"$ms$",'Color', '#ffffff','interpreter','latex', 'FontWeight','bold','FontSize',20);
%             text(10,200,"$Var: \ $"+num2str(var(motion_compensated_frame(:))),'Color', '#ffffff','interpreter','latex', 'FontWeight','bold','FontSize',20);
%             subtightplot(2,3,3);
%             plot(e.ts,e.vx,t,fieldx,'or','Markersize',8,'MarkerFaceColor', 'r','LineWidth',1);grid on;legend("Smoothed Flow","Current Flow")
%             text(t,fieldx+50,"$v_x: \ $"+num2str(fieldx)+"$[px/s]$",'FontSize',15,'Color', '#000000','interpreter','latex', 'FontWeight','bold');
%             subtightplot(2,3,6);
%             plot(e.ts,e.vy,t,fieldy,'or','Markersize',8,'MarkerFaceColor', 'r','LineWidth',1);grid on;legend("Smoothed Flow","Current Flow")
%             text(t,fieldy+50,"$v_y: \ $"+num2str(fieldy)+"$[px/s]$",'FontSize',15,'Color', '#000000','interpreter','latex', 'FontWeight','bold');
%             %         subtightplot(4,3,[7 12]);
%             %         imagesc(pixels.^(1/4));colormap(magma(100));axis on
%             %         c = colorbar;
%             %         c.LineWidth = 2;
%             %         c.FontSize = 20;c.Label.String = '$\sum events/px$';
%             %         c.Label.Interpreter = 'latex';
%             %         xlabel("$X [px]$",'interpreter','latex', 'FontWeight','bold','FontSize',20)
%             %         ylabel("$Y [px]$",'interpreter','latex', 'FontWeight','bold','FontSize',20)
%             %         text(20,100,"$Motion \ Compensated \ Map$",'Color', '#ffffff','interpreter','latex', 'FontWeight','bold','FontSize',20);
%             %         text(20,210,"$\Delta t: \ $"+num2str(events(end,1)/1e6)+"$s$",'Color', '#ffffff','interpreter','latex', 'FontWeight','bold','FontSize',20);
%             %         caxis([0 3])
            set(gcf,'Position',[50 1200 2200 500])
            drawnow
        end
        
        if RECORD
            F = getframe(gcf);
            writeVideo(writerObj, F);
        end
    end
end


if RECORD
    close(writerObj);
    fprintf('Sucessfully generated the video\n')
end

