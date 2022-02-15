# Mobile-Price-Range-Prediction

# Abstract

In this Modern Era, Smartphones are an integral part of the lives of human beings. When a smartphone is purchased ,many factors like the Display, Processor, Memory, Camera, Thickness, Battery, Connectivity and others are taken into account. One of the major factor which also people look is whether the phone they are buying is worth the cost. In the competitive mobile phone market companies want to understand sales data of mobile phones and factors which drive the prices. This project looks to solve the problem by taking the historical data pertaining to the key
features of smartphones along with its cost and develop a model that will predict the approximate price of
the new smartphone with a reasonable accuracy. The dataset used for this purpose has taken into consideration 21 different parameters for predicting the price of the phone. Random Forest Classifier,
Logistic Regression, Decision tree and KNN have been used primarily. Based on the accuracy, the
appropriate algorithm has been used to predict the prices of the smartphone. This not only helps the
customers decide the right phone to purchase, it also helps the owners decide what should be the appropriate
pricing of the phone for the features that they offer.

# Problem statement

In this project, we are going to explore and analyze a dataset which contains specifications of two thousand mobile phones and try to predict optimum price ranges for a list of mobile phones in the market by applying various machine learning algorithms such as logistic regression, decision tree, random forest and k-nearest neighbours(KNN).

# Packages Required

1. Numpy
2. Pandas
3. Matplotlib
4. Seaborn
5. Sklearn packages:
6. Train_test_split
7. Metrics
8. Logistic regression
9. Decision tree classifier
10. Random forest classifier
11. Kneighbours classifier
12. Classification report
13. Confusion matrix
14. Accuracy score
15. Gridsearch CV

# Approaches

--> After loading and reading the dataset in pandas Dataframe, we know that our dataset has 2000 rows and
21 columns.

--> We first imported all necessary libraries in our notebook which required for the analysis.

--> During Data exploration, we got to know that there are no Null values in the dataset.

--> The last attribute i.e., price_range column is a target variable. So our data have labels and we applied
supervised learning algorithms. We defined our target column as “Y” and rest of the data which are used
as inputs as “X”.

--> There are four price ranges as target values, so we did multiclass classification in our project.

--> Our dataset was perfectly balanced with 25% share to each type of price range.

--> During EDA, we got following insights from the data which helped us to identify which are the factors to
get a price range. Lets go through each EDA of columns:

• Battery Power- Low power batteries are slightly more in count

• Clock speed- Variance of clock speed is slightly more for mobiles in Category '0'

• Dual sim- Slightly more number of phones have dual sim. Price Range of dual sim phones are
considerably higher. This Denotes that Dual sim plays an important role in classification

• Fc - Front Camera mega pixels- price range and fc have less correlation

• Four G- Price Range of 4G phones are considerably higher. This Denotes that 4G plays an
important role in classification

• Mobile weight- Almost evenly spread across data set

• N_cores - Number of cores of processor- 67 mobiles in Price range of 0 is having 8 Cores

• Sc_h - Screen Height of mobile in cm- Some screen sizes are in high price range

• Sc_w - Screen Width of mobile in cm- Width ranges mostly in 0-7

• Three G- Price Range of 3G phones are considerably higher. This Denotes that 3G plays an important role in classification

• Touch Screen- Price Range of touch screen phone is low

• Wifi- Price Range of wifi phones are considerably higher. This Denotes that wifi plays an important role in classification

# Models

We used Logistic regression, Decision tree, Random forest and KNN algorithm for modelling. We
checked the accuracy score and confusion matrix of each algorithm.

**Logistic Regression:**

Logistic Regression is actually a classification algorithm that was given the name regression due to the fact that the mathematical formulation is very similar to linear regression.
The function used in Logistic Regression is sigmoid function or the logistic function given by:
		f(x)= 1/1+e ^(-x)

We got the accuracy of 73%, precision of 0.73, recall of 0.73 and F1 score of 0.73

**Decision Tree:**

Decision Tree is a Supervised learning technique that can be used for both classification and Regression problems, but mostly it is preferred for solving Classification problems. It is a tree-structured classifier, where internal nodes represent the features of a dataset, branches represent the decision rules and each leaf node represents the outcome. In a Decision tree, there are two nodes, which are the Decision Node and Leaf Node. Decision nodes are used to make any decision and have multiple branches, whereas Leaf nodes are the output of those decisions and do not contain any further branches. The decisions or the test are performed on the basis of features of the given dataset. It is a graphical representation for getting all the possible solutions to a problem/decision based on given conditions. 

We got the accuracy of 82%, precision of 0.83, recall of 0.83 and F1 score of 0.83

**Random Forest Classifier:**

Random Forest is a bagging type of Decision Tree Algorithm that creates a number of decision trees from a randomly selected subset of the training set, collects the labels from these subsets and then averages the final prediction depending on the most number of times a label has been predicted out of all.

We got the accuracy of 90%, precision of 0.90, recall of 0.90 and F1 score of 0.90

**K Nearest Neighbours:**

K-Nearest Neighbour is one of the simplest Machine Learning algorithms based on Supervised Learning technique. K-NN algorithm assumes the similarity between the new case/data and available cases and put the new case into the category that is most similar to the available categories. K-NN algorithm stores all the available data and classifies a new data point based on the similarity. This means when new data appears then it can be easily classified into a well suite category by using K- NN algorithm. K-NN algorithm can be used for Regression as well as for Classification but mostly it is used for the Classification problems. It is also called a lazy learner algorithm because it does not learn from the training set immediately instead it stores the dataset and at the time of classification, it performs an action on the dataset. KNN algorithm at the training phase just stores the dataset and when it gets new data, then it classifies that data into a category that is much similar to the new data.

We got the accuracy of 95%, precision of 0.95, recall of 0.95 and F1 score of 0.95

# Conclusion:

After training our dataset with four different model, we conclude that KNN is the best model for our

dataset (via the highest accuracy score of 0.95). The best optimum K number is to be 9 for this dataset.








