import os
import sys
import argparse


import numpy as np

from pylrpredictor.modelfactory import create_model, setup_model_combination,create_plotter



def cut_beginning(y, threshold=0.05, look_ahead=5):
    """
        we start at a point where we are bigger than the initial value for look_ahead steps
    """
    if len(y) < look_ahead:
        return y
    num_cut = 0
    for idx in range(len(y)-look_ahead):
        start_here = True
        for idx_ahead in range(idx, idx+look_ahead):
            if not (y[idx_ahead] - y[0] > threshold):
                start_here = False
        if start_here:
            num_cut = idx
            break
    return y[num_cut:]





class ProbabilisticExtrapolation:
    """
        Will evaluate p(y > y_best) and stop if the result doesn't look promising.
        In any other case we will continue running.
    """
        
    def __init__(self,xlim,models,nthreads=-1,predictive_std_threshold=None):
        self.predictive_std_threshold = predictive_std_threshold
        #models = ["vap", "ilog2", "weibull", "pow3", "pow4", "loglog_linear",
        #          "mmf", "janoschek", "dr_hill_zero_background", "log_power",
        #          "exp4"]
        self.xlim = xlim
        self.y=None
        model = setup_model_combination(#create_model(
            #"curve_combination",
            models=models,
            xlim=xlim,
            recency_weighting=True,
            nthreads=nthreads)
        self.model = model

    def predict(self):
        """
            predict f(x), returns 1 if not successful
        """
        return  self.model.predict(self.xlim)

    def plot(self): 
        plotter = create_plotter("curve_combination",self.model)
        plotter.posterior_plot(x=np.arange(1,1800))
        import matplotlib.pyplot as plt
        plt.show()


    def fit(self,data):
        self.y=np.array(data)
        #TODO subtract num_cut from xlim!
        self.y = cut_beginning(self.y)
        x = np.asarray(list(range(1, len(self.y)+1)))
        if not self.model.fit(x, self.y):
            #failed fitting... not cancelling
            print("failed fitting the model")
            return 0


    def check(self,ybest):
        assert (self.y is not None),"You should first fit the model"
        y_curr_best = np.max(self.y)
        if y_curr_best > ybest:
            #we already exceeded ybest ... let the other criterions decide when to stop
            print("Already better than ybest... not evaluating f(y)>f(y_best)")
            return 1

    def posterior_prob_x_greater_than(self,ybest):
        self.check(ybest)
        return self.model.posterior_prob_x_greater_than(self.xlim, ybest)

    def posterior_mean_prob_x_greater_than(self,ybest):
        self.check(ybest)
        return self.model.posterior_mean_prob_x_greater_than(self.xlim, ybest)


if __name__ == "__main__":

    data=np.loadtxt("learning_curve.txt") 
    models = ["vap", "ilog2", "weibull", "pow3", "pow4", "loglog_linear",
              "mmf", "janoschek", "dr_hill_zero_background", "log_power",
              "exp4"]
    term_crit = ProbabilisticExtrapolation(data.shape[0]*50,models,predictive_std_threshold=None)
    term_crit.fit(data)
    print(term_crit.posterior_prob_x_greater_than(0.70))
    print(term_crit.posterior_mean_prob_x_greater_than(0.70))
    print(term_crit.predict())


