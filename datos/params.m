% DATOS DEL MOTOR DE CONTINUA:
% (SON DATOS REALES DE UN MOTOR DEL LABORATORIO DE MANIOBRAS DE PERITOS)
Kp = 0.55;
Ri = 1.1648;
Li = 0.0068;
Kb = 0.82;
B  = 0.00776;
J  = 0.0271;

sim("motorcc.mdl")

writematrix([t,Vi,wm,Ii],'motor_dc.csv')