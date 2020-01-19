import numpy as np
import matplotlib.pyplot as plt

def prinzip(gamma, RM_list):
    """
    #---------------------------
    # Delta
    #---------------------------
    """
    p = np.linspace(0, 1, len(RM_list))
    phi = np.power(p, gamma)

    def make_delta_for_plot(liste, pop):
        new_val = [val for val in liste for _ in (0, 1)]
        if pop == -1:
            new_val.pop(-1)
        else:
            new_val.pop(0)
        return new_val

    new_x = make_delta_for_plot(p,pop=0)
    new_y = make_delta_for_plot(phi, pop=-1)

    plt.figure(figsize=(35,25)) 
    plt.subplot(221)
    plt.plot(p, phi)
    plt.plot(new_x,new_y)
    plt.xlabel('p')
    plt.ylabel('\u03C6(p)')
    plt.title(' ')
    plt.grid()
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    
    """
    #---------------------------
    # Fläche
    #---------------------------
    """
    b = gamma
    p = np.array(np.arange(0.001, 1, 1/len(RM_list)))
    p_liste = []
    for i in p:
        phi_p = b * i**(b-1)
        p_liste.append(phi_p)

    x = np.array(np.arange(0,1,1/len(RM_list)))
    liste = []
    for i in p_liste:
        liste.append(i)
    y = liste

    plt.subplot(222)
    plt.plot(p, p_liste, color='orange')
    plt.bar(x, liste, width=(1/len(RM_list)), edgecolor='black')
    plt.xlabel('p')
    plt.ylabel('\u03A6(p)')
    plt.title(' ')
    plt.grid()
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.show()