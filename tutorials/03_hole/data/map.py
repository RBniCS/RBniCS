from sympy import *
from numpy import *

# initialization
m1, m2  = symbols('m1 m2', real=True)


#####################
# subdomain number
subid = input('Enter subdomain id number: ')
#####################


if subid == 1:
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

if subid == 2:
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

if subid == 3:
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


if subid == 4:
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

if subid == 5:
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

if subid == 6:
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

if subid == 7:
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

if subid == 8:
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
# else:
#   print('wrong number of subdomain id')




K0 = Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

B = Matrix([[1, 0, x1, y1, 0, 0],
            [0, 1, 0, 0, x1, y1],
            [1, 0, x2, y2, 0, 0],
            [0, 1, 0, 0, x2, y2],
            [1, 0, x3, y3, 0, 0],
            [0, 1, 0, 0, x3, y3]])
 
V = Matrix([[xo_1,yo_1,xo_2,yo_2,xo_3,yo_3]])

V = Transpose(V)

A = Inverse(B)*V

#print A


C1 = A[0]
g11 = A[2]
g12 = A[3]
C2 = A[1]
g21 = A[4]
g22 = A[5]

print('C1 =', A[0])
print('g11 =', A[2])
print('g12 =', A[3])
print('C2 =', A[1])
print('g21 =', A[4])
print('g22 =', A[5])



G = Matrix([[g11, g12], [g21, g22]])
J = G.det()


D = Inverse(G)

Gg = Matrix([[D[0,0], D[0,1], 0], [D[1,0],D[1,1], 0], [0, 0, 1]])


K1 = J*Gg*K0*Transpose(Gg);

print('K11 =', simplify(K1[0,0]))
print('K22 =', simplify(K1[1,1]))
print('K33 =', simplify(K1[2,2]))
print('K12 =', simplify(K1[0,1]))
print('K21 =', simplify(K1[1,0]))
