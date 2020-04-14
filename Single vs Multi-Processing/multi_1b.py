#---------------------------------------------------------------------------------------------------------------
# Package Import
import pandas as pd
import numpy as np
from riskmeasure_module import risk_measure as rm
from Monte_Carlo_Simulation_lite import var_covar_matrix_func, var_func, cholesky_func, verteilung_func, copula_sim, hist_func

#-------------------------------------------------------------------------------------------------------------------------------------------
# Definiere die Funktion "repeat_parallel", welche die Simulation mit den jeweiligen Parameter ausführt
# Prinzipiell wie die for-Schleife im Standard-Script
def repeat_parallel(runs_sim, rand_x, rand_y, mu_list, std_list, corr_list, alpha, gamma, shared_list, VaR_list, CVaR_list, PSRM_list, i):

    _, _, _, total_summe_liste = copula_sim(runs_sim, rand_x, rand_y, mu_list, std_list, corr_list, full_log=False)
    shared_list += total_summe_liste
    x = rm(total_summe_liste, alpha, gamma)    
    VaR_list.append(x.VaR())    
    CVaR_list.append(x.CVaR())
    PSRM_list.append(x.Power())

#-------------------------------------------------------------------------------------------------------------------------------------------
# Definiere die Funktion "RM_frame_func", welche den DataFrame mit den Risikomaßen erstellt 
# und ausgibt und die dafür nötigen Argumente erhält; Genauere Beschreibung im Standard-Script
def RM_frame_func(runs_sim, runs_func, RM_VaR_list, RM_CVaR_list, RM_PSRM_list, SCREEN_WIDTH, centered):
    #---------------------------
    # Erzeuge ein DataFrame mit den Simulationsvergebnissen
    # und deren prozentualen Änderung vom jeweils vorherigen Ergebnis
    RM_frame = pd.DataFrame()
    RM_frame['VaR'] = RM_VaR_list
    RM_frame['VaR-Change'] = RM_frame['VaR'].pct_change()
    RM_frame['CVaR'] = RM_CVaR_list
    RM_frame['CVaR-Change'] = RM_frame['CVaR'].pct_change()
    RM_frame['Power'] = RM_PSRM_list
    RM_frame['Power-Change'] = RM_frame['Power'].pct_change()
    #---------------------------
    # Ermittle die kleinste und größte Relaisation des jweiligen Risikomaßes
    Min_Max_VaR = (min(RM_VaR_list), max(RM_VaR_list))
    Min_Max_CVaR = (min(RM_CVaR_list), max(RM_CVaR_list))
    Min_Max_PSRM = (min(RM_PSRM_list), max(RM_PSRM_list))
    #---------------------------
    # Gib den DataFrame und die Infos zurück
    print('#' + SCREEN_WIDTH * '-' + '#')
    print('|' + centered('[INFO] Der DataFrame mit den auf den auf ' +str(runs_func) + ' mal ' + str(runs_sim) + ' Durchläufen beruhenden Risikomaßen ergibt sich wie folgt: ') + '| ')
    print('#' + SCREEN_WIDTH * '-' + '#')
    print(RM_frame)
    print('#' + SCREEN_WIDTH * '-' + '#')
    print('|' + centered('Nach ' + str(runs_func) + ' Simulationsläufen mit je ' + str(runs_sim) + ' Durchläufen beträgt der kleinste VaR ' + str(round(Min_Max_VaR[0],2)) +', der größte ' + str(round(Min_Max_VaR[1],2)) + ' (\u0394 = ' + str((round((float(Min_Max_VaR[0]/Min_Max_VaR[1])-1)*100,2))) + '%).') + '| ')
    print('|' + centered('Nach ' + str(runs_func) + ' Simulationsläufen mit je ' + str(runs_sim) + ' Durchläufen beträgt der kleinste CVaR ' + str(round(Min_Max_CVaR[0],2)) +', der größte ' + str(round(Min_Max_CVaR[1],2)) + ' (\u0394 = ' + str((round((float(Min_Max_CVaR[0]/Min_Max_CVaR[1])-1)*100,2))) + '%).') + '| ')
    print('|' + centered('Nach ' + str(runs_func) + ' Simulationsläufen mit je ' + str(runs_sim) + ' Durchläufen beträgt das kleinste P-SRM ' + str(round(Min_Max_PSRM[0],2)) +', das größte ' + str(round(Min_Max_PSRM[1],2)) + ' (\u0394 = ' + str((round((float(Min_Max_PSRM[0]/Min_Max_PSRM[1])-1)*100,2))) + '%).') + '| ')
    print('#' + SCREEN_WIDTH * '-' + '#')

#-------------------------------------------------------------------------------------------------------------------------------------------
# Definiere die Funktion "plotty_func", welche das Plotten der Verteilungfunktionen übernimmt
#  und die dafür nötigen Argumente erhält; Genauere Beschreibung im Standard-Script
def plotty_func(runs_sim, runs_func, mega_summe_list):
    #-------------------------------------------
    # Zerlege die mega_summe_list (beinhaltet alle Ergenisse) in Teillisten,
    # welche die Ergebnisse der einzelnen Simulationsläufe beinhalten
    counter_0, counter_1 = 0, runs_sim
    array = []
    for _ in range(0, runs_func):
        x = mega_summe_list[counter_0:counter_1]
        array.append(x)
        counter_0 += runs_sim
        counter_1 += runs_sim
    #---------------------------
    # Erstelle für jedes Dieser Teillisten ein Histogramm
    # und plotte anschlißend das Ergebnis
    for items in array:
        values_PF = items
        bins = runs_sim    
        H, X1 = np.histogram(values_PF, bins, density=True)
        hist_func(H, X1)
    verteilung_func()