import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

for i in range (0, 25):
    lvel = []
    lp = []
    for j in range(0, 999):
        
        path = f"output/predictions_{i}_{j}.csv"
        
        df = pd.read_csv(path)
        
        u = df['u_pred']
        v = df['v_pred']    
        p = df['p_pred']
        
        vel = np.sqrt(u**2 + v**2)
        
        lvel.append(vel)        
        lp.append(p)
        
    
    plt.imshow(np.array(lvel), cmap = 'autumn')
    plt.colorbar()
    plt.title("vel")
    
    plt.show()
    
    plt.imshow(lp, cmap = 'autumn')
    plt.colorbar()
    plt.title("pressure")

    plt.show()
    
    
        
     
