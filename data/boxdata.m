n = 2000;
m = 2000;

R = rand(1,n);
A = [cos(2*pi*R);abs(sin(2*pi*R))]+ (rand(2,n)-.5)*.2;
R = rand(1,m);
B = [1.4*cos(2*pi*R);1.2*abs(sin(2*pi*R))] + (rand(2,m)-.5)*.2;
plot(A(1,:),A(2,:),'dr',B(1,:),B(2,:),'+b');

dlmwrite('box.data', [A',zeros(n,1);B',ones(m,1)]);
