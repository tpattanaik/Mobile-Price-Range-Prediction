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

We got the following evaluation metric for the algorithm we implemented:

 precision    recall  f1-score   support

           0       0.92      0.88      0.90       100
           1       0.72      0.64      0.68       100
           2       0.57      0.58      0.58       100
           3       0.72      0.82      0.77       100

    accuracy                           0.73       400
   macro avg       0.73      0.73      0.73       400
weighted avg       0.73      0.73      0.73       400


