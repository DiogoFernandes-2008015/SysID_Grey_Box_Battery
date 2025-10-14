This repository provides a collection of grey-box system identification scripts for battery modelling using equivalent circuit models (ECM). These models have linear state derivative equations and a nonlinear output equation due to the 9th degree polynomial relation between SOC and OCV.

The ECM used were the 1RC and the PNGV (3RC) models.

For each model, MATLAB and Python scripts were developed. The MATLAB implementation is based on the nonlinear grey-box estimation function, and the Python script is a JAX and Diffrax-based implementation of the grey-box optimization problem with a multiple shooting strategy.
 
- 1 RC Model Equation:

States: $x = [SOC, V_0]$

Input: $u=i$

State derivative equation:

$\dot{SOC} = -\frac{un}{C}$

$\dot{V}_1 = -\frac{V_1}{R_1C_1}+\frac{u}{C_1}$

Output equation:

$y = OCV(SOC)+R_0u+V_1$

with:
$OCV = \sum_{i=1}^{9} a_kSOC^k$

Parameters to be tuned: $R_0,R_1,C_1,n$

- PNGV model:

States: $x = [SOC, V_0,V_1,V_2,V_3]$

Input: $u=i$

State derivative equation:

$\dot{SOC} = -\frac{un}{C}$

$\dot{V}_0 = \frac{u}{C_0}$

$\dot{V}_1 = -\frac{V_1}{R_1C_1}+\frac{u}{C_1}$

$\dot{V}_2 = -\frac{V_2}{R_2C_2}+\frac{u}{C_2}$

Output equation:

$y = OCV(SOC)+R_0u+V_0+V_1+V_2$

with:
$OCV = \sum_{i=1}^{9} a_kSOC^k$

Parameters to be tuned: $R_0,C_0,R_1,C_1,R_2,C_2,n$

Data provided by 

This code is adapted from code made available by  , introducing the capacity to tune models with more than one state and nonlinear outputs. 
