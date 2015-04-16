clear all
clc

% initialization
syms m1 m2 real;


%%%%%%%%%%%%%%%%%%%%%
% subdomain number
subid = input('Enter subdomain id number: ');
%%%%%%%%%%%%%%%%%%%%%

switch subid
case 1
	x1 = 1;
	y1 = -1;
	x2 = -1;
	y2 = -1;
	x3 = -2;
	y3 = -2;
	
	xo_1 = m1;
	yo_1 = -m2;
	xo_2 = -m1;
	yo_2 = -m2;
	xo_3 = x3;
	yo_3 = y3;

case 2
	x1 = -2;
	y1 = -2;
	x2 = 2;
	y2 = -2;
	x3 = 1;
	y3 = -1;
	
	xo_1 = x1;
	yo_1 = y1;
	xo_2 = x2;
	yo_2 = y2;
	xo_3 = m1;
	yo_3 = -m2;

case 3
	x1 = -2;
	y1 = -2;
	x2 = -1;
	y2 = -1;
	x3 = -1;
	y3 = 1;
	
	xo_1 = x1;
	yo_1 = y1;
	xo_2 = -m1;
	yo_2 = -m2;
	xo_3 = -m1;
	yo_3 = m2;


case 4
	x1 = -2;
	y1 = -2;
	x2 = -1;
	y2 = 1;
	x3 = -2;
	y3 = 2;
	
	xo_1 = x1;
	yo_1 = y1;
	xo_2 = -m1;
	yo_2 = m2;
	xo_3 = x3;
	yo_3 = y3;

case 5
	x1 = 2;
	y1 = -2;
	x2 = 1;
	y2 = 1;
	x3 = 1;
	y3 = -1;
	
	xo_1 = x1;
	yo_1 = y1;
	xo_2 = m1;
	yo_2 = m2;
	xo_3 = m1;
	yo_3 = -m2;

case 6
	x1 = 2;
	y1 = -2;
	x2 = 2;
	y2 = 2;
	x3 = 1;
	y3 = 1;
	
	xo_1 = x1;
	yo_1 = y1;
	xo_2 = x2;
	yo_2 = y2;
	xo_3 = m1;
	yo_3 = m2;

case 7
	x1 = -1;
	y1 = 1;
	x2 = 1;
	y2 = 1;
	x3 = -2;
	y3 = 2;
	
	xo_1 = -m1;
	yo_1 = m2;
	xo_2 = m1;
	yo_2 = m2;
	xo_3 = x3;
	yo_3 = y3;

case 8
	x1 = 1;
	y1 = 1;
	x2 = 2;
	y2 = 2;
	x3 = -2;
	y3 = 2;
	
	xo_1 = m1;
	yo_1 = m2;
	xo_2 = x2;
	yo_2 = y2;
	xo_3 = x3;
	yo_3 = y3;
otherwise
	error('wrong number of subdomain id');
end




K0 = [1 0 0; 0 1 0; 0 0 1];

B = [1 0 x1 y1 0 0;
     0 1 0 0 x1 y1;
     1 0 x2 y2 0 0;
     0 1 0 0 x2 y2;
     1 0 x3 y3 0 0;
     0 1 0 0 x3 y3];
 
V = [xo_1;yo_1;xo_2;yo_2;xo_3;yo_3];
 
A = inv(B)*V;

C1 = A(1);
g11 = A(3);
g12 = A(4);
C2 = A(2);
g21 = A(5);
g22 = A(6);




G = [g11 g12;
      g21 g22];
J = det(G);

D = inv(G);

Gg = [D(1,:) 0; D(2,:) 0; 0 0 1];

K1 = J*Gg*K0*Gg';

K11 = simplify(K1(1,1))
K22 = simplify(K1(2,2))
K33 = simplify(K1(3,3))
K12 = simplify(K1(1,2))
K21 = simplify(K1(2,1))


 
