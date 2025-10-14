Code for system identification of an equivalent circuit model of a battery based on a 1 RC branch circuit using a JAX-based code in Python for implementing the optimization problem.

The code uses a multiple-shooting strategy to optimize its solution.

States: $x = [SOC, V_0]$

Input: $u=i$

State derivative equation:

$\dot{SOC} = -\frac{un}{C}$

$\dot{V}_1 = -\frac{V_1}{R_1C_1}+\frac{u}{C_1}$

Output equation:

$y = OCV(SOC)+R_0u+V_1$

with:
$OCV = \sum_{i=1}^{9} a_kSOC^k$

Parameters to be tuned: $R_0,R_!,C_1,n$

Data provided by 

This code is adapted from code made available by 
