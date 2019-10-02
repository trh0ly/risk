import numpy as np
import matplotlib.pyplot as plt

def prinzip(gamma, RM_list):
    """
    #---------------------------
    # Delta
    #---------------------------
    """
    p = np.linspace(1/len(RM_list), len(RM_list)/len(RM_list), len(RM_list))
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
    p = np.array(np.arange(1/len(RM_list), len(RM_list)/len(RM_list), 1/len(RM_list)))
    phi_p = np.power((b * p), (b-1))
    x = np.array(np.arange(1/len(RM_list),len(RM_list)/len(RM_list),1/len(RM_list)))

    liste = []
    for i in phi_p:
        liste.append(i)
    y = liste

    plt.subplot(222)
    plt.plot(p, phi_p, color='orange')
    plt.bar(x, liste, width=(1/253), edgecolor='black')
    plt.xlabel('p')
    plt.ylabel('\u03A6(p)')
    plt.title(' ')
    plt.grid()
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.show()