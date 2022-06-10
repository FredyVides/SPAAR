"""
Created on Fri Apr 29 19:30:41 2022
Lorenz system simulator
@author: Fredy Vides
"""

def NLSystem2(m = 3):
    from numpy import arange
    from scipy.integrate import odeint
    
    def System(state, t, m):
        x, Dx = state
        
        dx = Dx
        dDx = -x + m*(1-x**2)*Dx
        
        return [dx, dDx]
    
    p = (m,)
    y0 = [-4, 2.0]
    t = arange(0.0, 150.0, 0.03)
    
    r = odeint(System, y0, t, p)
    
    return t,r