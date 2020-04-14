#---------------------------------------------------------------------------------------------------------------
# Package Import
import scipy.stats as st 
from scipy.stats import rankdata, norm  
from scipy import array, linalg, dot
import random 
import numpy as np 
import math
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
from riskmeasure_module import risk_measure as rm

#---------------------------------------------------------------------------------------------------------------
# Hilfsfunktionen 
#-------------------------------------------------------------
# Definition einer Funktion, welche eine Varianz-Kovarianz-Matrix erstellt
# Argumente:
# - std_list: Liste mit Standardabweichungen
# - corr_list: Liste mit Korrelationskoeffizienten
#------------
def var_covar_matrix_func(std_list, corr_list):    
    counter_0, counter_1 = 0, 0
    len_std_list = len(std_list)
    array = [[0] * len_std_list] * len_std_list    
    val_list = []
    
    # Für jedes i und j in len_std_list..
    for i in range(0,len_std_list):
        for j in range(0,len_std_list):
            # Wenn i = j, dann multipliziere beide Werte
            if i == j:
                val = std_list[i] * std_list[i]
                val_list.append(val)
            # Wenn i kleiner j, dann multipliziere beide Standardabweichungen
            # und die dazugehörige Korrelation
            if i < j:
                val = (std_list[i] * std_list[j] * corr_list[counter_0])
                counter_0 += 1
                val_list.append(val)
            # Wenn i größer j, dann multipliziere beide Standardabweichungen
            # und die dazugehörige Korrelation
            if i > j:
                val = (std_list[i] * std_list[j] * corr_list[counter_1])
                counter_1 += 1
                val_list.append(val)
                
    var_covar = np.array(val_list).reshape(len_std_list, len_std_list)      
    return var_covar
    
#-------------------------------------------------------------
# Definition einer Funktion, welche ein Varianz-Array erstellt
# Argumente:
# - std_list: Liste mit Standardabweichungen
#------------
def var_func(std_list):    
    var_list = []    
    for i in range(0, len(std_list)):
        var = np.power(std_list[i],2)
        var_list.append(var)        
    return var_list

#-------------------------------------------------------------
# Definition einer Funktion, welche die Cholesky-Zerlegung durchführt
# Argumente:
# - matrix: Varianz-Kovarianz-Array
#------------
def cholesky_func(matrix): 
    
    # Leere Matrix anlegen
    lower = [[0 for x in range(len(matrix))]  
                for y in range(len(matrix))]
  
    # Zerlegt eine Matrix in unter Dreiecksmatrix
    # und deren transponierte Matrix
    for i in range(len(matrix)):  
        for j in range(i + 1):  
            sum1 = 0  
            # Diagonale
            if j == i:  
                for k in range(j): 
                    sum1 += pow(lower[j][k], 2) 
                lower[j][j] = int(math.sqrt(matrix[j][j] - sum1)) 
            # Nicht-Diagonale    
            else:                   
                # Bestimmung L(i, j) durch Nutzung Formel für L(j, j) 
                for k in range(j): 
                    sum1 += (lower[i][k] * lower[j][k])
                if(lower[j][j] > 0): 
                    lower[i][j] = int((matrix[i][j] - sum1) / lower[j][j]) 
                    
    # Transformation in ein Numpy-Array und Rückgabe
    lower = np.array(lower)
    return lower

#-------------------------------------------------------------
# Hilfsfunktionen zum Plotten

#---------------------------       
# Definition einer Funktion, ein Histogramm plottet
#------------        
def hist_func(H, X1):
    dx = X1[1] - X1[0]
    F1 = np.cumsum(H) * dx
    plt.plot(X1[1:], F1)

#---------------------------
# Definition einer Funktion, welche eine Verteilungsfunktion plottet
#------------    
def verteilung_func():
    plt.title('Verteilung X+Y, Realisationen gleichverteilte X,Y mit Gauss-Copula')
    blue_patch = mpatches.Patch(color='blue', label='Monte-Carlo-Simulation(en)')
    plt.legend(handles=[blue_patch], loc='upper left')
    plt.grid()
    plt.xlabel('PF-Realisation')
    plt.ylabel('Wahrscheinlichkeit')
    left, right = plt.xlim()
    plt.xlim((left, right))
    plt.xlim(left, right)
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.show()

#---------------------------------------------------------------------------------------------------------------
# Definition einer Funktion, welche die Monte-Carlo-Sumulation durchführt
# Argumente:
# - simulation_runs: Anzahl an Durchläufen der Simulation
# - randverteilung_x: Neue Randverteilung der Variablen X
# - randverteilung_y: Neue Randverteilung der Variablen Y
# - mu_list: Liste mit Erwartungswerten
# - std_list: Liste mit Standardabweichungen
# - corr_list: Liste mit Korrelationen
# - m: Anzahl der Variablen die pro Lauf gezogen werden (hier gerade nur zwei möglich)
# - full_log:
# --> Wenn True: Alle berechneten Größen werden den Listen angefügt
# --> Wenn False: Nur die Summe (das letztendliche Ergebnis der Simulation) der Realisationen wird einer Liste angefügt
#------------
def copula_sim(simulation_runs, randverteilung_x, randverteilung_y, mu_list, std_list, corr_list, m=2, full_log=False):
    
    # Listen für die zu berechnenden Größen
    total_standard_norm_ab_list, total_realisation_cop_list = [], []
    total_xy_list, total_summe_liste = [], []
    
    # Funktionen aufrufen um die für die folgenden Berechnungen nötige Werte zu erhalten
    var_covar = var_covar_matrix_func(std_list, corr_list)
    var_list = var_func(std_list)
    cholesky = cholesky_func(var_covar)
    
    # Durchführung der n Durchläufe
    for i in range(0, simulation_runs):        
        #---------------------------------------------------------------------------------------------------------------
        # Gleichverteilte Zufallsvariablen ziehen        
        random_ZV_list = []  
        for i in range(0, m):
            x = random.random()
            random_ZV_list.append(x)
        if simulation_runs == 1:
            print('1) Gleichverteilte Zufallsvariablen: {}\n'.format(random_ZV_list))
        
        #---------------------------------------------------------------------------------------------------------------
        # Transformation der gleichverteilten Zufallsvariablen in unabhängige standardnormalverteilte Zufallsvariablen
        standard_norm_list = norm.ppf(random_ZV_list)   
        if simulation_runs == 1:
            print('2) Standardnormalverteilte Zufallsvariablen: {}\n'.format(standard_norm_list))
        
        #---------------------------------------------------------------------------------------------------------------
        # Transformation in standardnormalverteilte abhängige Zufallsvariablen        
        standard_norm_ab_list = []
        counter_cholesky_0, counter_cholesky_1 = 0, 0
        counter_mu, counter_standard_norm_list = 0, 0

        # Für jede zuvor gezogene unabhängige, standardnormalverteilte Zufallsvariable 
        # wird die innere Abhänigkeitsstruktur (aus der Cholesky-Matrix) auf diese Zufallsvariable übertragen
        for i in range(0, m):
            # 1. Schritt: Berechnung a, mit a = 1. unabhängige standardnormalverteilte Zufallsvariable * Eintrag in Cholesky-Matrix
            a = cholesky[counter_cholesky_0][counter_cholesky_1] * standard_norm_list[counter_standard_norm_list]
            counter_cholesky_1 += 1
            counter_standard_norm_list += 1

            # 2. Schritt: Berechnung b, mit b = 2. unabhängige standardnormalverteilte Zufallsvariable * Eintrag in Cholesky-Matrix
            b = cholesky[counter_cholesky_0][counter_cholesky_1] * standard_norm_list[counter_standard_norm_list]
            counter_cholesky_0 += 1  
            counter_cholesky_1 = 0
            counter_standard_norm_list = 0  

            # 3. Schritt: Endergebnis r ist der Erwartungswert + a + b und stellt die abhängige standardnormalverteilte Zufallsvariable dar
            r = a + b + mu_list[counter_mu]
            standard_norm_ab_list.append(r)
            counter_mu += 1
        if simulation_runs == 1:
            print('3) Standardnormalverteilte abhängige Zufallsvariablen: {}\n'.format(standard_norm_ab_list))

        #---------------------------------------------------------------------------------------------------------------
        # Transformation Realisationen der Gauss-Copula
        counter = 0
        realisation_cop_list = []     
        # Transformation indem von jeweiliger abhängiger standardnormalverteilter Zufallsvariablen
        # der dazugehörige Erwartungswert subtrahiert wird, und dieses Ergebnis dann durch die dazugehörige Standardabweichung geteilt wird
        # Die Realisation der Gauss-Copula ergibt sich dann als der Wert (Wahrscheinlichkeit) der kumulierten Normalverteilungsfunktion
        for i in range(0, m):
            r_cop = (standard_norm_ab_list[counter] - mu_list[counter]) / math.sqrt(var_list[counter])   
            p_value = st.norm.cdf(r_cop)
            realisation_cop_list.append(p_value)
            counter += 1
        if simulation_runs == 1:
            print('4) Realisationen der Gauss-Copula: {}\n'.format(realisation_cop_list))
            
        #---------------------------------------------------------------------------------------------------------------
        # Gemeinsame Verteilung: Übertragung der neuen Ränder auf die ermittelte Abhängigkeitsstruktur  
        # Für x und y wird jeweils (Obergrenze - Untergrenze) * Realisation der Gauss-Copula + Untergrenze gerechnet
        # Dabei sind Obergrenze bzw. Untergrenze der linke bzw. rechte Rand des Intervalls der neuen Randverteilungen
        x = (randverteilung_x[0] - randverteilung_x[1]) * realisation_cop_list[0] + randverteilung_x[1]
        y = (randverteilung_y[0] - randverteilung_y[1]) * realisation_cop_list[1] + randverteilung_y[1]
        
        # Die Summe bzw. das Endergebnis der Simulation ist dann X + Y, also die gemeinsame Realisation
        summe = x + y 
        if simulation_runs == 1:
            print('5) Sumulationsergebnisse: x={}, y={}, Summe={}'.format(x, y, summe))
        
        # Berechnete Werte werden den jeweiligen Listen angefügt
        if full_log == True:
            total_standard_norm_ab_list.append(standard_norm_ab_list)
            total_realisation_cop_list.append(realisation_cop_list)
            total_xy_list.append((x, y))
        total_summe_liste.append(summe)        
  
    return total_standard_norm_ab_list, total_realisation_cop_list, total_xy_list, total_summe_liste

#---------------------------------------------------------------------------------------------------------------
# Testlauf
if __name__=='__main__':
    # Anzahl Simulationsdurchläufe
    n = 10000
    # Neue Randverteilungen (Gleichverteilung)
    rand_x = [10,20]
    rand_y = [8,22]
    # Varianzen und Korrelation(en)
    var_x = 4 
    var_y = 9
    corr_list = [0]
    std_list = [math.sqrt(var_x), math.sqrt(var_y)]
    # Erwartungswerte
    mu = [2, 3]
   
    _, _, _, _ = copula_sim(1, rand_x, rand_y, mu, std_list, corr_list, full_log=False)