% Plotting weights
% Last modified: July, 2015
% Author: Junfeng Wen (junfeng.wen@ualberta.ca), University of Alberta
function plotWeight(x1,x2,y1,y2,weight,f)

    minX = min([x1;x2]);
    maxX = max([x1;x2]);
    x = (minX-0.1):0.05:(maxX+0.1);
    y = f(x);
    fg = figure;
    hold on
    plot(x1, y1, 'r+');
    plot(x2, y2, 'bo');
    [AX,H1,H2] = plotyy(x,y,x1,weight,...
        @(a,b)plot(a,b,'k-'),@(a,b)bar(a,b,1,'m','EdgeColor','m'));
    set(get(AX(1),'Ylabel'),'String','Output');
    set(get(AX(2),'Ylabel'),'String','Weight');
    xlabel('Input');
    set(AX(1),'YLim',[-1,9]);
    set(AX(1),'YTick',[-1:1:9]);
    set(AX(2),'YLim',[0,0.5]);
    set(AX(2),'YTick',[0:0.05:0.5]);
    hl = legend('Train point','Test point','True model','Weight','Location','Northwest');
    set(findall(fg,'type','text'),'fontSize',18);
    
    hLegendPatch = findobj(hl, 'type', 'patch');
    v = get(hLegendPatch, 'vertices');
    Length = (v(2,2) - v(2,1))/20; % the desire length
    Middle = (v(2,2)+v(2,1))/2;
    v(1,2) = Middle - Length/2;
    v(2,2) = Middle + Length/2;
    v(3,2) = Middle + Length/2;
    v(4,2) = Middle - Length/2;
    v(5,2) = Middle - Length/2;
    set(hLegendPatch, 'vertices', v);

% END OF FUNCTION
end
