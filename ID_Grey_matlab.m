%Code for nonlinear grey model identification based on the matlab toolbox
%Diogo Lopes Fernandes
%The model must be defined on an auxliary function defined as
%dx/dt=F(t,u,x)
%y=H(t,u,x)
%[dx, y]=model(t,x,u,p1,p2,...,pN,FileArgument)
%dx -> derivative of the states of the model (F(t,x,u))
%y -> output of the model (H(t,x,u))
%p1,...,pN -> parameters to be estimated

clear all
close all
clc

%%      
%Load experimental data
DATA = load("data_train.mat");
for j=1:1:length(DATA.i)
    t(j,1)=j;
end
data = iddata(DATA.v,DATA.i,1,'Name','Battery');
data.InputName = 'Current';
data.InputUnit = 'A';
data.OutputName = 'Terminal Voltage';
data.OutputUnit = 'V';
data.Domain = 'Time';
figure
idplot(data)

%%
%Modelo 1RC
%Construction o the idnlgrey object
Ny = 1; %number of outputs
Nx = 2; %number of states
Nu = 1; %number of inputs
Order = [Ny Nu Nx];

%Description of the parameters
Parameters(1).Name = 'R0';%name of the parameter
Parameters(1).Unit = 'Ohm';%Unit of the parameter
Parameters(1).Value =rand();%initial estimative for the parameter
Parameters(1).Minimum = 0.01;%minimum value for the parameter
Parameters(1).Maximum = 1;%maximum value for the parameter
Parameters(1).Fixed = 0;
Parameters(2).Name = 'R1';%name of the parameter
Parameters(2).Unit = 'Ohm';%Unit of the parameter
Parameters(2).Value = 1+99*rand();%initial estimative for the parameter
Parameters(2).Minimum = 1;%minimum value for the parameter
Parameters(2).Maximum = 100;%maximum value for the parameter
Parameters(2).Fixed = 0;
Parameters(3).Name = 'C1';%name of the parameter
Parameters(3).Unit = 'Faraday';%Unit of the parameter
Parameters(3).Value = 100+9900*rand();%initial estimative for the parameter
Parameters(3).Minimum = 100;%minimum value for the parameter
Parameters(3).Maximum = 10000;%maximum value for the parameter
Parameters(3).Fixed = 0;
Parameters(4).Name = 'n';%name of the parameter
Parameters(4).Unit = '';%Unit of the parameter
Parameters(4).Value = 1;%initial estimative for the parameter
Parameters(4).Minimum = 0.0001;%minimum value for the parameter
Parameters(4).Maximum = 1;%maximum value for the parameter
Parameters(4).Fixed = 0;
%Initial Condition of the model
InitialStates = [0.98;0];

%Handle to the model function
func_handle = @battery_1nrc;

nlgrmod = idnlgrey(func_handle, Order, Parameters, InitialStates);
nlgrmod.Ts = 0;
nlgrmod.InputName = 'Current';
nlgrmod.InputUnit = 'A';
nlgrmod.OutputName = 'Terminal Voltage';
nlgrmod.OutputUnit = 'V';

nlgrmod

%%
%Estimation phase
%Configurations of the estimations algorithim
nlgropt = nlgreyestOptions;
nlgropt.Display = 'on';
nlgropt.SearchOptions.MaxIterations = 10000;
nlgropt.SearchOptions.FunctionTolerance = 1e-6;
nlgropt.SearchOptions.StepTolerance = 1e-6;

%Estimaton command
nlgrmod = nlgreyest(data, nlgrmod, nlgropt)

%Results
nlgrmod.Report.Fit
nlgrmod.Report.Parameters.ParVector
nlgrmod.Report.Termination
mod1RC = nlgrmod
%%
%Results analysis
%Comparing data and estimated data
figure
compare(data, nlgrmod)
figure
resid(data,nlgrmod)

%%
%Comparing results with validation data set
DATAval = load("data_val.mat");
for j=1:1:length(DATAval.i)
    t(j,1)=j;
end
dataval = iddata(DATAval.v,DATAval.i,1,'Name','Battery');
dataval.InputName = 'Current';
dataval.InputUnit = 'A';
dataval.OutputName = 'Terminal Voltage';
dataval.OutputUnit = 'V';
dataval.Domain = 'Time';
figure
idplot(dataval)
figure
compare(dataval, nlgrmod)
grid
figure
resid(dataval,nlgrmod)
grid

%%
% %Modelo PNGV
% %Construction o the idnlgrey object
% Ny = 1; %number of outputs
% Nx = 4; %number of states
% Nu = 1; %number of inputs
% Order = [Ny Nu Nx];
% 
% %Description of the parameters
% Parameters(1).Name = 'R0';%name of the parameter
% Parameters(1).Unit = 'Ohm';%Unit of the parameter
% Parameters(1).Value =0.99*rand()+0.01;%initial estimative for the parameter
% Parameters(1).Minimum = 0.01;%minimum value for the parameter
% Parameters(1).Maximum = 1;%maximum value for the parameter
% Parameters(1).Fixed = 0;
% Parameters(2).Name = 'C0';%name of the parameter
% Parameters(2).Unit = 'Faraday';%Unit of the parameter
% Parameters(2).Value = 100+9900*rand();%initial estimative for the parameter
% Parameters(2).Minimum = 100;%minimum value for the parameter
% Parameters(2).Maximum = 10000;%maximum value for the parameter
% Parameters(2).Fixed = 0;
% Parameters(3).Name = 'R1';%name of the parameter
% Parameters(3).Unit = 'Ohm';%Unit of the parameter
% Parameters(3).Value = 1+99*rand();%initial estimative for the parameter
% Parameters(3).Minimum = 1;%minimum value for the parameter
% Parameters(3).Maximum = 100;%maximum value for the parameter
% Parameters(3).Fixed = 0;
% Parameters(4).Name = 'C1';%name of the parameter
% Parameters(4).Unit = 'Faraday';%Unit of the parameter
% Parameters(4).Value = 100+9900*rand();%initial estimative for the parameter
% Parameters(4).Minimum = 1;%minimum value for the parameter
% Parameters(4).Maximum = 10000;%maximum value for the parameter
% Parameters(4).Fixed = 0;
% Parameters(5).Name = 'R2';%name of the parameter
% Parameters(5).Unit = 'Ohm';%Unit of the parameter
% Parameters(5).Value = 1+99*rand();%initial estimative for the parameter
% Parameters(5).Minimum = 1;%minimum value for the parameter
% Parameters(5).Maximum = 100;%maximum value for the parameter
% Parameters(5).Fixed = 0;
% Parameters(6).Name = 'C2';%name of the parameter
% Parameters(6).Unit = 'Faraday';%Unit of the parameter
% Parameters(6).Value = 100+9900*rand();%initial estimative for the parameter
% Parameters(6).Minimum = 100;%minimum value for the parameter
% Parameters(6).Maximum = 10000;%maximum value for the parameter
% Parameters(6).Fixed = 0;
% Parameters(7).Name = 'n';%name of the parameter
% Parameters(7).Unit = '';%Unit of the parameter
% Parameters(7).Value = rand();%initial estimative for the parameter
% Parameters(7).Minimum = 0;%minimum value for the parameter
% Parameters(7).Maximum = 1;%maximum value for the parameter
% Parameters(7).Fixed = 0;
% 
% %Initial Condition of the model
% InitialStates = [0.98;0;0;0];
% 
% %Handle to the model function
% func_handle = @battery_pngv_n;
% 
% nlgrmod = idnlgrey(func_handle, Order, Parameters, InitialStates);
% nlgrmod.Ts = 0;
% nlgrmod.InputName = 'Current';
% nlgrmod.InputUnit = 'A';
% nlgrmod.OutputName = 'Terminal Voltage';
% nlgrmod.OutputUnit = 'V';
% 
% nlgrmod

%%
% %Estimation phase
% %Configurations of the estimations algorithim
% nlgropt = nlgreyestOptions;
% nlgropt.Display = 'on';
% nlgropt.SearchOptions.MaxIterations = 10000;
% nlgropt.SearchOptions.FunctionTolerance = 1e-6;
% nlgropt.SearchOptions.StepTolerance = 1e-6;
% 
% 
% %Estimaton command
% nlgrmod = nlgreyest(data, nlgrmod, nlgropt)
% modpngv = nlgrmod
% 
% %Results
% nlgrmod.Report.Fit
% nlgrmod.Report.Parameters.ParVector
% nlgrmod.Report.Termination
% 
% %%
% %Results analysis
% %Comparing data and estimated data
% figure
% compare(data, nlgrmod)
% figure
% resid(data,nlgrmod)
% 
% %%
% %Comparing results with validation data set
% DATAval = load("data_val.mat");
% for j=1:1:length(DATAval.i)
%     t(j,1)=j;
% end
% dataval = iddata(DATAval.v,DATAval.i,1,'Name','Battery');
% dataval.InputName = 'Current';
% dataval.InputUnit = 'A';
% dataval.OutputName = 'Terminal Voltage';
% dataval.OutputUnit = 'V';
% dataval.Domain = 'Time';
% figure
% idplot(dataval)
% figure
% compare(dataval, nlgrmod)
% grid
% figure
% resid(dataval,nlgrmod)
% grid
% 
% %%
% %Saving the models
save('modelo_pngv_best','modpngv')
