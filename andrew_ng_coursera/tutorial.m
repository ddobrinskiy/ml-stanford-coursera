for i=1:10,
  v(i) = 2^i;
end;

while true,  % break statements
  v(i) = 999;
  i = i+1;
  if i ==6,
    break;
  end;
end;

if v(1)==1,
    disp('The value is one');
  elseif v(1)==2,
    disp('the value is two');
  else
    disp('the value is not one or two');
end;



% Octave search path (advanced)
% addpath('/home/user/Dropbox/ml/octaveFunctions')
