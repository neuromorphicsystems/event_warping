function e = event1d(w,ax,d,deltat,signal,noise,np,fig)

lambda0 = noise/deltat;
lambda1 = signal/deltat;

backSpike = rand(1,deltat) < lambda0;
starSpike = rand(1,deltat) < lambda1;
centrex_start = 1;centrex_end = 20;
vx_gt = (centrex_end - centrex_start) / deltat;

% rng(0);
e = []; idx = 0;
for t = 1:deltat
    vx_gt = (centrex_end - centrex_start + ax*t) / deltat;
    starX = t*vx_gt + centrex_start;
    
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
    
    if starSpike(t)
        idx = idx+1;
        px = randi(d,1) + starX;
        if px < w
            x = px;
        else
            x=nan; 
        end
        e.x(idx) = round(x+randn*np);
        e.y(idx) = 1;
        e.t(idx) = t;
        e.l(idx,1)=1;
        e.l(idx,2)=0;
        e.vxgt(idx) = vx_gt*1e5;
    end
end
if fig
    figure(344565);
    plot(e.x(e.l(:,1)==0),e.t(e.l(:,1)==0)/1e6,'.b');hold on
    plot(e.x(e.l(:,1)>0),e.t(e.l(:,1)>0)/1e6,'.r','MarkerSize',10);
    xlabel('$X [px]$','interpreter','latex', 'FontWeight','bold','FontSize',20)
    ylabel('$t \ (s)$','interpreter','latex', 'FontWeight','bold','FontSize',20)
    title(num2str(w) + "px 1D sensor",'interpreter','latex', 'FontWeight','bold','FontSize',20)
    legend("Sky","Star");
end
end

