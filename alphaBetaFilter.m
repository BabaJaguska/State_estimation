function [est_x, est_y, rateOfChange_x, rateOfChange_y] = smoothOneTarget(prev_x, prev_y,curr_x, curr_y, alpha_x, beta_x, alpha_y, beta_y, rateOfChange_x, rateOfChange_y)


%prediction
x_pred = prev_x + rateOfChange_x; % * 1 frame
y_pred = prev_y + rateOfChange_y;
%rateOfChange = rateOfChange;

%update
residual_x = curr_x - x_pred;
residual_y = curr_y - y_pred;

est_x = x_pred + alpha_x*residual_x;
est_y = y_pred + alpha_y*residual_y;

rateOfChange_x = rateOfChange_x + beta_x*residual_x; %per frame
rateOfChange_y = rateOfChange_y + beta_y*residual_y; %per frame

if est_x<0
    est_x = 0;
end

if est_y<0
    est_y = 0;
end



end
