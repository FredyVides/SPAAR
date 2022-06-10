"""
Created on Fri Apr 29 19:30:41 2022
Lorenz system simulator
@author: Fredy Vides
"""

def NLSystem1(m = 16, k = 1, d = -25):
    from numpy import arange
    from scipy.integrate import odeint
    
    def System(state, t, m, k, d):
        x, Dx = state
        
        dx = Dx
        dDx = (1/m)*(-k*x+d*x**3)
        
        return [dx, dDx]
    
    p = (m,k,d)
    y0 = [1.0, 0.0]
    t = arange(0.0, 40.0, 0.01)
    
    r = odeint(System, y0, t, p)
    
    return t,r