function pixels = weight1d(pixels,fieldx,deltat,width)

t  = deltat/1e6;
vx = abs(fieldx*t);
x  = repmat(1:size(pixels,2),[size(pixels,1),1]);

if vx > 0 && vx < width
    index = (x > 0) & (x < vx);
    pixels(index) = (pixels(index)).*((vx)./x(index));
    index = (x > width) & (x < width+vx);
    pixels(index) = (pixels(index)).*((vx)./(-x(index)+width+vx));
end

if vx > width && vx < width+vx
    index = (x > 0) & (x < width);
    pixels(index) = (pixels(index)).*((vx)./x(index));
    midP = ((vx)./x(index));
    index = (x >= width) & (x < vx);
    pixels(index) = (pixels(index)).*midP(end);
    index = (x > vx) & (x < width+vx);
    pixels(index) = (pixels(index)).*((vx)./(-x(index)+width+vx));
end
pixels(1:5)=[];pixels(end-5:end)=[];
% pixels(1:ceil((t/1)))=[];pixels(end-ceil((t/1)):end)=[];
end

