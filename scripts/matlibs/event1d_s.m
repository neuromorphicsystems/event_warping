function e = event1d_s(w,ax,d,deltat,signal,noise,np,fig)
ax_down     = -9e-7;
lambda0     = noise/deltat;
lambda1     = signal/deltat;

backSpike = rand(1,deltat) < lambda0;
starSpike = rand(1,deltat) < lambda1;
centrex_start = 10;centrex_end = 49;
vx_gt = (centrex_end - centrex_start) / deltat;

% rng(0);
e = []; idx = 0;
for t = 1:deltat
    
    if backSpike(t)
        idx = idx+1;
        x = randi(w,1);
        e.x(idx) = round(x);
        e.y(idx) = 1;
        e.t(idx) = t;
        e.l(idx,1)=0;
        e.l(idx,2)=0;
        e.vxgt(idx) = vx_gt*1e5;
    end
    
    % acceleration
    if starSpike(t) && t < 20e6
        ax = 0;
        vx_gt = (centrex_end - centrex_start + ax*t) / deltat;
        starX = t*vx_gt + centrex_start;
        idx = idx+1;
        px = randi(d,1) + starX;
        if px < w
            x = px;
        else
            x=nan; 
        end
        e.x(idx) = round(x+(randn*np));
        e.y(idx) = 1;
        e.t(idx) = t;
        e.l(idx,1)=1;
        e.l(idx,2)=0;
        e.vxgt(idx) = vx_gt*1e5;
        e.axgt(idx) = ax*1e5;
    end
    
    % decceleration
    if starSpike(t) && t > 20e6 && t < 40e6
        vx_gt = (centrex_end - centrex_start + ax_down*t) / deltat;
        starX = t*vx_gt + centrex_start;
        idx = idx+1;
        px = randi(d,1) + starX;
        if px < w
            x = px;
        else
            x=nan;
        end
        e.x(idx) = round(x+(randn*np))+8;
        e.y(idx) = 1;
        e.t(idx) = t;
        e.l(idx,1)=1;
        e.l(idx,2)=0;
        e.vxgt(idx) = vx_gt*1e5;
        e.axgt(idx) = ax_down*1e5;
    end
    
    % acceleration
    if starSpike(t) && t > 40e6 && t < 60e6
        ax = 1e-7;
        vx_gt = (centrex_end - centrex_start + ax*t) / deltat;
        starX = t*vx_gt + centrex_start;
        idx = idx+1;
        px = randi(d,1) + starX;
        if px < w
            x = px;
        else
            x=nan;
        end
        e.x(idx) = round(x)-14;
        e.y(idx) = 1;
        e.t(idx) = t;
        e.l(idx,1)=1;
        e.l(idx,2)=0;
        e.vxgt(idx) = vx_gt*1e5;
        e.axgt(idx) = ax*1e5;
    end
end
if fig
    figure(344565);
    plot(e.x(e.l(:,1)==0),e.t(e.l(:,1)==0)/1e6,'.b',e.x(e.l(:,1)>0),e.t(e.l(:,1)>0)/1e6,'.r','MarkerSize',10);hold on
    xlabel('$X [px]$','interpreter','latex', 'FontWeight','bold','FontSize',20)
    ylabel('$t \ (s)$','interpreter','latex', 'FontWeight','bold','FontSize',20)
    title(num2str(w) + "px 1D sensor",'interpreter','latex', 'FontWeight','bold','FontSize',20);grid on;xlim([1 w])
%     subplot(3,1,3)
%     plot(e.vxgt,"LineWidth",2);grid on;
%     xlabel('$t \ (s)$','interpreter','latex', 'FontWeight','bold','FontSize',15)
%     ylabel('$v_x [px/x]$','interpreter','latex', 'FontWeight','bold','FontSize',15)
%     set(gcf,'Position',[50 1200 600 700]);
end
end

