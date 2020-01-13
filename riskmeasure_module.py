"""
#################################################################################
# Risk Measure Module                                                           #
# © Thomas Robert Holy 2019                                                     #
# Version 0.1.0                                                                 #
# E-Mail: th0ly96[at]gmail.com                                                  #
#################################################################################
"""
#--------------------------------------------------------------------------------
# Package import

import numpy as np

#--------------------------------------------------------------------------------
# Define class risk_measure

class risk_measure:

    def __init__(self, data, alpha=0.1, gamma=0.5):
        self.data = sorted(data)
        self.alpha = alpha
        self.gamma = gamma
        self.expected_value = np.mean(self.data)

    """
    Function to determine Value at Risk of Dataset;
    Given: Dataset, Alpha-Quantile
    Output: Value at Risk at Alpha-Quantile
    """
    def VaR(self):
        item = (int((self.alpha * len(self.data))) - 1)
        self.VaR = -(self.data[item])
        return self.VaR

    """
    Function to determine Conidtional Value at Risk of Dataset
    Given: Dataset, Alpha-Quantile
    Output: Conditional Value at Risk from Beginn of the Dataset
    to Alpha-Quantile
    """
    def CVaR(self):
        item = int((self.alpha * len(self.data))) 
        CVaR_list = self.data[0:item] 
        self.CVaR = -(np.sum(CVaR_list) / len(CVaR_list)) 
        return self.CVaR

    """
    Function to determine Power Spectral Risk Measure of Dataset
    Given: Dataset, Gamma
    Output: Power Sepctral Risk Measure and Expected Value of Dataset
    """
    def Power(self):        
        subj_ws_list = [] 
        counter_1, counter_2 = len(self.data), (len(self.data) - 1)
        for _ in self.data:
            subj_ws = (np.power((counter_1 / len(self.data)), self.gamma)) - (np.power((counter_2 / len(self.data)), self.gamma))
            counter_1 -= 1 
            counter_2 -= 1 
            subj_ws_list.append(subj_ws) 
        subj_ws_list = subj_ws_list[::-1] 
        self.power = (np.matmul(np.transpose(self.data), subj_ws_list))
        return self.power

    """
    Function to determine standard deviation of Dataset
    Given: Dataset
    Output: Standard Deviation of Dataset
    """
    def std(self):
        self.std = np.std(self.data)
        return self.std

    """
    Function to determine Variance of Dataset
    Given: Dataset
    Output: Variance of Dataset
    """
    def var(self):
        self.var = np.var(self.data)
        return self.var

#----------------------------------------------------------------------------------

def test():
    sum_list = [33.71557708879494, 32.89040605540057, 29.960258802464732, 23.879146368931536, 37.3527604890338, 36.39501750600361, 28.148859214934664, 32.44820678736961,
                39.064138806942395, 35.513264429935305, 31.973463411796548, 20.51711878560922, 35.83228014751211, 26.847736219158826, 29.061280204137674, 28.575379461381573,
                35.318629613623756, 29.12591606654684, 35.352748935194356, 24.27105248975043, 28.18837919963555, 36.63959749240546, 28.55272818032561, 25.113120534924178,
                37.50770282488068, 32.87467159486174, 31.203771872112654, 31.05737562650088, 34.73062371587045, 23.489427626775345, 35.74291959900605, 36.60107657095012,
                29.611577956651836, 29.597943635070713, 34.907986873778405, 21.02720478867177, 34.080558490627354, 28.890895160135848, 25.400351593402753, 40.13490746729601,
                30.911422576441346, 25.305704855388413, 34.862588724863464, 35.656990156429536, 33.959608638899354, 24.06542617335804, 34.077177606512834, 36.19894865510238,
                27.685515026034984, 35.72684592964698, 33.638700065706026, 32.705380195623356, 31.438880998500643, 27.52092660179834, 24.96088408450973, 20.37301712866975,
                19.157230210889793, 26.688318539816215, 23.510796243850788, 37.29563366111959, 35.23274192240539, 32.94106900042442, 22.782632381362195, 23.861749660311524,
                24.670774226279796, 31.519399401256322, 24.684258472604256, 26.99502270721643, 31.08409700304391, 31.640716920951405, 26.724149192576554, 25.772809832969788,
                32.746225469692156, 33.02892846324279, 30.54837050837699, 30.688217551789013, 27.87112171351845, 31.325295739198758, 31.012202476014973, 22.352421446776333,
                25.177396124557916, 31.373326202641543, 23.204207818572094, 31.2383550517498, 38.337413657137915, 33.34792308924565, 30.296662640280637, 31.968247896588917,
                30.091664634516327, 29.840185315752912, 32.04827943031411, 35.15211967504939, 20.15504739996002, 34.960572909188215, 26.910806628424798, 19.620890151352842,
                39.03980489876725, 30.95984233433436, 38.091317550360685, 26.324500543698583]

    x = risk_measure(sum_list, alpha=0.1, gamma=0.5)
    print('Der Erwartungswert der gegebenen Liste beträgt: {}'.format(x.expected_value))
    print('Die Standardabweichung obiger Liste beträgt: {}'.format(x.std()))
    print('Die Varianz obiger Liste beträgt: {}\n'.format(x.var()))
    print('Der Value at Risk der gegebenen Liste beträgt: {}'.format(x.VaR()))
    print('Der Conditional Value at Risk der gegebenen Liste beträgt: {}'.format(x.CVaR()))
    print('Das Risiko des Power-Spektralen Risikomaß nach obiger Liste beträgt: {}'.format(x.Power()))

#----------------------------------------------------------------------------------
if __name__=='__main__': test()
