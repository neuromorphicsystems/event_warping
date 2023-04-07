function [variance,pixels] = maximise_variance(events,vx,w,weight)
deltat   = events.t(end) - events.t(1);
warpedx  = round(events.x+vx*events.t/1e6);
pixels   = accumulate(warpedx,events.y);
if weight
    pixels  = weight1d(pixels,vx,deltat,w);
end
variance = var(pixels(:));
end

