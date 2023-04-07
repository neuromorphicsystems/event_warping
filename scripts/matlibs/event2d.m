function e = event2d(w,h,ax,ay,d,deltat,signal,noise,fig)

lambda0 = noise/deltat;
lambda1 = signal/deltat;

backSpike = rand(1,deltat) < lambda0;
starSpike = rand(1,deltat) < lambda1;
centrex_start = 1;centrex_end = 1;
centrey_start = 5;centrey_end = 5;
vx_gt = (centrex_end - centrex_start) / deltat;
timeWindow   = 2e5;

rng(0);
e = []; idx = 0;
for t = 1:deltat
    vx_gt = (centrex_end - centrex_start + ax*t) / deltat;
    vy_gt = (centrey_end - centrey_start + ay*t) / deltat;
    starX = t*vx_gt + centrex_start;
    starY = t*vy_gt + centrey_start;
    if backSpike(t)
        idx = idx+1;
        x = randi(w,1);
        y = randi(h,1);
        e.x(idx) = round(x);
        e.y(idx) = round(y);
        e.ts(idx) = t;
        e.l(idx,1)=0;
        e.l(idx,2)=0;
        e.vxgt(idx) = vx_gt*1e5;
    end
    
    if starSpike(t)
        idx = idx+1;
        px = randi(d,1) + starX;
        py = randi(d,1) + starY;
        if px < w
            x = px;
        else
            x=nan;
        end
        if py < h
            y = py;
        else
            y=nan;
        end
        
        e.x(idx) = round(x);
        e.y(idx) = round(y);
        e.ts(idx) = t;
        e.l(idx,1)=1;
        e.l(idx,2)=0;
        e.vxgt(idx) = vx_gt*1e5;
    end
end
e.x = e.x';
e.y = e.y';
e.ts = e.ts';
if fig
    figure(12323556);
    plot3(e.x(e.l(:,1)==0),e.y(e.l(:,1)==0),e.ts(e.l(:,1)==0)/1e6,'.b');hold on
    plot3(e.x(e.l(:,1)>0),e.y(e.l(:,1)>0),e.ts(e.l(:,1)>0)/1e6,'.r','MarkerSize',10);
    xlabel('$X \ [px]$','interpreter','latex', 'FontWeight','bold','FontSize',20)
    xlabel('$Y \ [px]$','interpreter','latex', 'FontWeight','bold','FontSize',20)
    zlabel('$t \ (s)$','interpreter','latex', 'FontWeight','bold','FontSize',20)
    title(num2str(w) + "x" + num2str(h) + "px 2D sensor",'interpreter','latex', 'FontWeight','bold','FontSize',20)
    legend("Sky","Star");
end

end

