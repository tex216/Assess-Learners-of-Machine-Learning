""""""
"""  		  	   		   	 		  		  		    	 		 		   		 		  
Test a learner.  (c) 2015 Tucker Balch  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		   	 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		   	 		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Template code for CS 7646  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		   	 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		   	 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		   	 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		   	 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		   	 		  		  		    	 		 		   		 		  
or edited.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		   	 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		   	 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		   	 		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		   	 		  		  		    	 		 		   		 		  
"""

import math
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)
    data = np.genfromtxt(sys.argv[1], delimiter=',')
    data = data[1:, 1:]  # remove date and header

    # compute how much of the data is training and testing
    # 60% training (In-sample), 40% testing (Out-of-sample)
    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data  		  	   		   	 		  		  		    	 		 		   		 		  
    train_x = data[:train_rows, 0:-1]
    train_y = data[:train_rows, -1]
    test_x = data[train_rows:, 0:-1]
    test_y = data[train_rows:, -1]

    ### Q1: DTLearner testing for leaf_size 1-20
    rmse_in_sample = []
    rmse_out_sample = []
    for i in range(1, 21):
        # create a learner and train it
        learner = dt.DTLearner(leaf_size=i, verbose=True)
        learner.add_evidence(train_x, train_y)
        # evaluate in sample
        pred_y = learner.query(train_x)
        rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
        rmse_in_sample.append(rmse)
        # evaluate out of sample
        pred_y2 = learner.query(test_x)
        rmse2 = math.sqrt(((test_y - pred_y2) ** 2).sum() / test_y.shape[0])
        rmse_out_sample.append(rmse2)
    # plotting the figure
    plt.plot(rmse_in_sample)
    plt.plot(rmse_out_sample)
    plt.title("RMSE vs Leaf Size for DTLearner")
    plt.xlabel("Leaf Size")
    plt.xlim(1, 20)
    plt.ylabel("RMSE")
    plt.ylim(0, 0.01)
    plt.legend(["In-Sample", "Out-Sample"])
    plt.savefig("Q1.png")
    plt.close("all")

    ### Q2: BagLearner testing for leaf_size 1-20: using DTLearner for 10 bags
    rmse_in_sample = []
    rmse_out_sample = []
    for i in range(1, 21):
        # create a learner and train it
        learner = bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size": i}, bags=10, boost=False, verbose=False)
        learner.add_evidence(train_x, train_y)
        # evaluate in sample
        pred_y = learner.query(train_x)
        rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
        rmse_in_sample.append(rmse)
        # evaluate out of sample
        pred_y2 = learner.query(test_x)
        rmse2 = math.sqrt(((test_y - pred_y2) ** 2).sum() / test_y.shape[0])
        rmse_out_sample.append(rmse2)
    # plotting the figure
    plt.plot(rmse_in_sample)
    plt.plot(rmse_out_sample)
    plt.title("RMSE vs Leaf Size for BagLearner with 10 Bags")
    plt.xlabel("Leaf Size")
    plt.xlim(1, 20)
    plt.ylabel("RMSE")
    plt.ylim(0, 0.01)
    plt.legend(["In-Sample", "Out-Sample"])
    plt.savefig("Q2.png")
    plt.close("all")

    ### Q3: DTLearner vs RTLearner
    # a) measuring training time for both with leaf_size 1-20
    time_dt = []
    time_rt = []
    # create a DTlearner and train it
    for i in range(1, 21):
        start = time.time()
        learner = dt.DTLearner(leaf_size=i, verbose=True)
        learner.add_evidence(train_x, train_y)
        end = time.time()
        time_dt.append(end - start)
    # create a RTlearner and train it
    for j in range(1, 21):
        start2 = time.time()
        learner2 = rt.RTLearner(leaf_size=j, verbose=True)
        learner2.add_evidence(train_x, train_y)
        end2 = time.time()
        time_rt.append(end2 - start2)

    # plotting the figure
    plt.plot(time_dt)
    plt.plot(time_rt)
    plt.title("Training Time for DTLearner vs RTLearner")
    plt.xlabel("Leaf Size")
    plt.xlim(1, 20)
    plt.ylabel("Time (s)")
    plt.ylim(0, 2)
    plt.legend(["DTLearner", "RTLearner"])
    plt.savefig("Q3a.png")
    plt.close("all")

    # b) measuring Mean Absolute Error (MAE) for training both with leaf_size 1-20
    mae_dt = []
    mae_rt = []
    for i in range(1, 21):
        # create a DTlearner and train it
        learner = dt.DTLearner(leaf_size=i, verbose=True)
        learner.add_evidence(train_x, train_y)
        # create a RTlearner and train it
        learner2 = rt.RTLearner(leaf_size=i, verbose=True)
        learner2.add_evidence(train_x, train_y)
        # evaluate DTlearner in sample
        pred_y = learner.query(train_x)
        pred_y = np.array(pred_y)
        train_y = np.array(train_y)
        mae = np.mean(np.abs(train_y - pred_y)) * 100
        mae_dt.append(mae)
        # evaluate RTlearner in sample
        pred_y2 = learner2.query(train_x)
        pred_y2 = np.array(pred_y2)
        mae2 = np.mean(np.abs(train_y - pred_y2)) * 100
        mae_rt.append(mae2)

    # plotting the figure
    plt.plot(mae_dt)
    plt.plot(mae_rt)
    plt.title("MAE for Training DTLearner vs RTLearner")
    plt.xlabel("Leaf Size")
    plt.xlim(1, 20)
    plt.ylabel("MAE")
    plt.ylim(0, 0.8)
    plt.legend(["DTLearner", "RTLearner"])
    plt.savefig("Q3b.png")
    plt.close("all")
