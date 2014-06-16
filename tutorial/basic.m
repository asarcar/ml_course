% ------------------------------
% background: Prompt Format Help
% ------------------------------
PS1('> '); % change prompt to '> '
format short; % choose short vs long format
display(pi); % display and disp are not same cmds
disp(sprintf('2 decimals: %0.2f', pi)); % display 2 decimal digits
addpath('utils');
help disp; % help data on "disp" or any cmd
help help; % displays information about help
% ------------------------------
% Basic
% ------------------------------
a = 1, b = 2, c = 3; % chain cmds: last cmd with ";" not printed
a = 4; b = 5; c = 6; % mult cmds in one line; none printed
A = ones(2, 3); % all ones 2x3 matrix
v = 3*zeros(1, 3); % all zero 1x3 row vector
B = rand(3, 3); % 3x3 random vector between 0 & 1
I = eye(3); % 3x3 Identity Matrix
C = -5 + 2*randn(1, 10000); % produces a row vector of 10,000 elements
			    % with mean -5 and std dev -2
hist(C, 100); % produces histogram binning data into 100 buckets
nA = size(A); % return a 1x2 row vector: [#rows #cols]
ndims(A) % return dimensions
rows(A) % # of rows
columns(A) % # of columns
% ------------------------------
% Loading, Saving, Scope
% ------------------------------
pwd % pwd, ls, cd: commands work as expected
load featuresX.dat; % loads the table into featuresX variable
load priceY.dat;
who % variables in the current scope
clear ans % clears the variable ans from current scope
v=priceY(1:4) % slice of priceY vector
save hello.mat v; % saves vector v and misc information
save hello.txt v -ascii % save vector v as text
% ------------------------------
% Matrix Modifications
% ------------------------------
A = [1 2; 3 4; 5 6];
w = A(:, 2); % ":" means everything along that row/col
X = A([1, 3], :); % assign X row 1 and 3 of A
A = [A, [100; 101; 102]]; % append another col to right of A
W = A(:); % transform A into a single vector with rows traversed first
C = [A A]; % concatenate A with itself row wise (side by side)
D = [C; C]; % concatenate A with itself col wise (top to bottom)
A(:, 1) = [10; 11; 12]; % modifies 1st col of A
C(2, 2) = 8; % change a specific element of matrix C
% ------------------------------
% Operations & Computations
% ------------------------------
A = [1 2; 3 4; 5 6];
B = [10 11; 20 21; 30 31];
C = [1 1; 2 2];
A * C
A .* B % .* is a pairwise operation
% Map Operations: Element wise
% ------------------------------
A .^ 2 % 
log(A) % 
exp(A)
1 ./ A
A + 1
A < 3
find( A < 3)
floor(A)
ceil(A)
abs(A - 3*ones(3, 2))
max(A, [], 2) % max of A along row (dim 2) rather than col (dim 1)
max(A, 4*rand(3,2))
% Result Vectors: Extracting Matrix Operation Results
% ------------------------------
[v_val v_ind] = max(A) % value and index vector for each col of A
[r c] = find(A >= 7) % return row and col vector of elements >= 7
% Reduce Operations
% ------------------------------
sum(A)
prod(A)
max(max(A, [], 2)) % max among all rows of A
max(A(:)) % max of A when vectorized
% ------------------------------
% Matrix Transformations
% ------------------------------
A' % A Transpose
flipud(A) % reflect A from a horizontal line in middle
fliplr(A) % reflect A from a vertical line in middle
% Validation Example: Validate Magic Matrix
A = magic(9); % matrix with same row, col, diag sum
prod(sum(A, 1) == (sum(A, 2))') %sum row == sum col
sum(sum(A .* eye(9))) == sum(A(1, :)) % sum diagonal == sum row
sum(sum(A .* fliplr(eye(9)))) == sum(A(1, :)) % sum diagonal == row
% Inverse
% ------------------------------
% pseudo inverse same as inverve for invertible matrix
round(pinv(A)*A*100000) == eye(9)*100000
round(pinv(A)*100000) == round(inv(A)*100000) 
% ------------------------------
% Plotting Tools
% ------------------------------
t = [0:0.01:0.98]; % Create a row vector from 0, 0.01, 0.02,...,0.98
y1 = sin(2*pi*4*t);
plot(t, y1);
y2 = cos(2*pi*4*t);
plot(t, y2);
% Plot one over other
plot(t, y1); hold on; plot(t, y2, 'r'); 
xlabel('Time'); ylabel('Value'); 
legend('sin', 'cos'); title('my plot');
print dpng 'myPlot.png'; % save plot in png format
close;
% Create multiple figures
figure(1); plot(t, y1); figure(2); plot(t, y2, 'r');
close; close;
subplot(1,2,1); % Divide plot in 1x2 grid: access 1st element
plot(t, y1);
subplot(1,2,2); plot(t, y2, 'r');
axis([0.5 1 -1 1]); % x axis (0.5 1) & y axis (-1 1)
clf; % clear figure: Frame kept open
% Color Maps: create a 5x5 grid of colors with color legend 
imagesc(magic(5)), colorbar, colormap gray;
close;
% ------------------------------
% Control Statement: for, while, if
% ------------------------------
v = zeros(10,1);
for i=1:10, v(i) = 2^i; end; % control block: till "end"
indices = 1:10; % row vector: 1 to 10
for i=indices; disp(sprintf("2 decimals: %0.2f", v(i))); end;
i=1; 
while true, 
  v(i) = 100+i;
  if (i>=6), 
    break;
  elseif (i<=2),
    v(i) = v(i) + 1;
  else
    v(i) = -v(i);
  end;
  i=i+1;
end;
% ------------------------------
% Function Invocation
% ------------------------------
addpath('utils'); % call non-local dir defined fns
[a b] = squareAndCube(10) % function call example
% Linear Cost Function
X = [1 1; 1 2; 1 3]; % Example training input data
y = [1; 2; 3]; % Example training output data
theta = [1; 0]; % Predicated Coefficients
addpath('utils'); % call non-local dir defined fns
j = linearCostFn(X, y, theta);
theta = [0; 1];
j2 = linearCostFn(X, y, theta);
