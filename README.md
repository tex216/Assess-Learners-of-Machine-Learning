# Assess-Learners-of-Machine-Learning

All files were coded in Python 3, including 1). A classic Decision Tree learner based on JR Quinlan algorithm; 2). A Random Tree learner based on A Cutler algorithm; 3). A Bootstrap Aggregating (Bagging) learner ensembled different learners; 4). An Insane leaner used specific use-case of the Bagging learner. Given the same data set, the differences and performance of these learners will be compared, thoroughly discussed, and evaluated by detailed analysis.

## Files：

1. DTLearner.py

Contains the code for the regression Decision Tree class to train and query a Decision Tree Learner. 

2. RTLearner.py

Contains the code for the regression Random Tree class to train and query a Random Tree Learner. 

3. BagLearner.py

Contains the code for the regression Bag Learner (i.e., a BagLearner containing Random Trees) to train and query with a learner ensemble.

4. InsaneLearner.py

Contains the code for the regression Insane Learner of Bag Learners.  

5. LinRegLearner.py

Contains the code for the regression Linear Learner.

6. testlearner.py

This file is considered the entry point to this project. All the experimental plots and testing statistics for the report could be generated once running this file.


## How To Run:    
                                                                           
PYTHONPATH=../:. python testproject.py Data/Istanbul.csv 
Details in README_Report.pdf