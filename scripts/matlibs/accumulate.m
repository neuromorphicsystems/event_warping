function pixels = accumulate_render(warpedx,warpedy)
sigma = 0.0;
padding = double(ceil(0.0 * sigma));
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

for i = 1:numel(xis)
    sumF = zeros(size(kernel_indices,1),size(kernel_indices,1),size(kernel_indices,3));
    for l = 1:size(kernel_indices,1)
        for j = 1:size(kernel_indices,1)
            summation = 0;
            for axis  = 1:size(kernel_indices,3)
                summation = summation + kernel_indices(l,j,axis);
                sumF(l,j,axis) = summation;
            end
        end
    end
    sumF = sumF.^ 0.0;
    finalSummation = (sumF(:,:,1) + sumF(:,:,2))*sigma_factor;
%     pixels(yis(i)-padding+1:yis(i)+padding+1,xis(i)-padding+1:xis(i)+padding+1) = pixels(yis(i)-padding+1:yis(i)+padding+1,xis(i)-padding+1:xis(i)+padding+1)+exp(finalSummation);
    pixels(yis(i)-padding+1:yis(i)+padding+1,xis(i)-padding+1:xis(i)+padding+1) = pixels(yis(i)-padding+1:yis(i)+padding+1,xis(i)-padding+1:xis(i)+padding+1)+1;

end
end

