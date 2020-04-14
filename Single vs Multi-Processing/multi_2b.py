import math
from multiprocessing import Process
from multiprocessing import Manager
from multiprocessing.pool import Pool
import datetime as dt
import operator
from multi_1b import repeat_parallel, RM_frame_func, plotty_func

#------------------------------------------------------------------------------------------------
# Definiere die Funktion "run" in welcher die Parallelisierung der Monte-Carlo-Simulation auf "Multi_1b" gemanaget wird
def run(runs_func, runs_sim, rand_x, rand_y, mu, std_list, corr_list, alpha, gamma, draw=False, SCREEN_WIDTH=100):
    RM_VaR_list = []
    RM_CVaR_list = []
    RM_PSRM_list = []
    mega_summe_list = []

    centered=operator.methodcaller('center', SCREEN_WIDTH)

    start = dt.datetime.now() # Startpunkt Zeitmessung
    with Manager() as manager: # Verwendung Funktion "Manager()" aus "multiprocessing" als Manager der geteilten Listen zwischen den parallel laufenden Simulationen
        shared_list = manager.list() # Legt die leere Liste "shared_list" an, welche zwischen den parallel laufenden Simulationen geteilt wird und alle Realisationen der einzelnen Simulationsläufe enthält
        VaR_list = manager.list() # Legt die leere Liste "VaR_list" an, welche zwischen den parallel laufenden Simulationen geteilt wird und alle VaR enthält
        CVaR_list = manager.list() # Legt die leere Liste "CVaR_list" an, welche zwischen den parallel laufenden Simulationen geteilt wird und alle CVaR enthält
        PSRM_list = manager.list() # Legt die leere Liste "PSRM_list" an, welche zwischen den parallel laufenden Simulationen geteilt wird und alle P-SRM enthält
        processes = [] # Legt die leere Liste "processes" an, in welcher die auszuführenden Prozesse abgelegt werden
        # Für jedes i in der Range 0 bis runs_func (Simulationsläufe)...
        for i in range(runs_func):
            p = Process(target=repeat_parallel, args=(runs_sim, rand_x, rand_y, mu, std_list, corr_list, alpha, gamma, shared_list, VaR_list, CVaR_list, PSRM_list, i)) # Erstelle den Prozess "p", welcher die Funktion "repeat_parallel" ausführt mit den gebenen Paramtertn
            p.start() # Starte den Prozess
            processes.append(p) # Füge den Prozess "p" der Liste "processes" an
        # Für jenden Prozess "p" in "processes" führe due Funktion join() aus (Clean Exit Process)
        for p in processes:
            p.join()
        RM_VaR_list += VaR_list # Füge der "RM_VaR_list" den jeweiligen VaR an
        RM_CVaR_list += CVaR_list # s.o.
        RM_PSRM_list += PSRM_list # s.o.
        # Sofern draw == 'True' füge die "shared_list" in der "mega_summe_list" an
        if draw == True:
            mega_summe_list += shared_list
    end = dt.datetime.now() # nedpunkt für die Zeitmessung
    RM_frame_func(runs_sim, runs_func, RM_VaR_list, RM_CVaR_list, RM_PSRM_list, SCREEN_WIDTH, centered) # Ausführen der Funktion, welche den DataFrame mit den Risikomaßen ausgibt
    # Sofern draw == 'True' führe die Funktion aus, welche die Verteilungfunktionen der Simulationsläufe plottet
    if draw == True:
        plotty_func(runs_sim, runs_func, mega_summe_list)
    print(end-start) # Gibt das Zeit-Delta zurück

#------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    #-------------------------------------------
    # Parameter für die Simulation
    #------------------------------
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
    #------------------------------
    #-------------------------------------------
    # Paramaeter Risikomaße
    alpha = 0.05
    gamma = 0.5
    #-------------------------------------------
    # Anzahl Simulationsläufe und Durchläufe pro Simulation
    runs_func = 10
    runs_sim = 1000

    run(runs_func, runs_sim, rand_x, rand_y, mu, std_list, corr_list, alpha, gamma, draw=False)