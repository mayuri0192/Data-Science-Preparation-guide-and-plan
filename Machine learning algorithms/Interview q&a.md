**What is multicolinearity?**
[youtube video 1](https://www.youtube.com/watch?v=sVJW5UXe84s),
[youtube video 2](https://www.youtube.com/watch?v=Cba9LJ9lS8s)

Example dataset: 
|cgpa | IQ  |backlogs | salary |
|-----|------|--------|--------|
|3.2 |120 | 5 | 20000|

cgpa, IQ, backlogs - dependent variables
salary - dependent variables. There are two types - +ve and -ve colinearity

negative colinearity : IQ -goes up backlogs - goes down

```salary = w0 + w1 * IQ + w2 * cgpa```

*w1* represents changes in salary when when IQ is changed considering cgpa is constant
*w2* represents changes in salary when when cgpa is changed considering IQ is constant

```Regression analysis is when independent variables are able to explain the variance in the dependent variable. Multicolinearity breaks this assumption. ```

Multicoloinearity is present when IQ <-> cgpa

It is difficult to figure out influence of independent variabels on dependent variables because of multicolinearity. 

Not a big issue for just predicting values
Issues: 
- Not suitable for feature importance. 

Types of Multicolinearity 

Structural - When data scientists introduce colinearity between features. Example: 

|City|
|----|
|Mumbai|
|Pune|
|Bangalore|

| Mumbai | Pune | Bangalore |
|-----|------ |------|
|0|1|0|

- data based - when colinearity is presnt in data. 

X & X^2

**Multicolinearity can be found from:**
- ```sns.heat(df.corr())```. if corr > ~0.9-0.95 - its a problem
- Variance Inflation Factor (VIF) analysis
if vif factor > 5 - remove one of the feature

[<img align="center" src="/Machine learning algorithms/assets/vif.PNG" width="600"/>](/Machine learning algorithms/assets/vif.PNG)
- Add more data - helps in getting rid of collinearity
- Apply Lasso/Ridge Regression - It automatically shrinks non important coeffs
- You can use PCA - which gives unrelated predictors

**Is multicolinearity bad for all ML algorithms:**

- It affects parametric algorithms:
    - Parametric - Linear Regr, Naive baised which basically assumes dependent and independent variables

    - Non parametric - Tree based algorithms. 

**Feature scaling**

Feature scaling is required in machine learning to ensure that the numerical features used in your model are on a similar scale. It's a preprocessing step that transforms the features so that they have comparable magnitudes. Feature scaling is important for several reasons:

1. Gradient Descent Convergence:
Many machine learning algorithms, particularly those that involve optimization techniques like gradient descent, perform better when features are on a similar scale. This is because algorithms converge faster and more reliably when the feature values are balanced.

2. Weight Sensitivity:
Algorithms like linear regression and support vector machines assign weights to features during training. If features have different scales, the algorithm might assign disproportionately higher importance to features with larger magnitudes. Scaling ensures that all features contribute equally.

3. K-Means Clustering:
In clustering algorithms like K-means, the distance between data points is a crucial factor. If features have different scales, the clustering algorithm might be heavily influenced by features with larger ranges, leading to biased results.

4. Regularization:
Regularization techniques, such as L1 and L2 regularization, penalize large coefficients. Feature scaling helps ensure that regularization works uniformly across all features, preventing some features from dominating the regularization process.

5. Distance-Based Algorithms:
Algorithms that rely on distance calculations, like K-nearest neighbors (KNN), are sensitive to feature scales. Features with larger scales can dominate the distance calculations, leading to inaccurate results.

6. Neural Networks:
Feature scaling is also beneficial in neural networks. It helps improve the training efficiency and convergence rate, making it easier for the network to find the optimal weights.

Common Feature Scaling Techniques:
Two common methods of feature scaling are:

Min-Max Scaling (Normalization): Scales features to a specific range (e.g., [0, 1]). It subtracts the minimum value from each feature and then divides by the range (max-min).

Standardization: Transforms features to have a mean of 0 and a standard deviation of 1. It subtracts the mean from each feature and then divides by the standard deviation.

Choosing the appropriate scaling method depends on the algorithm you're using and the characteristics of your data. Feature scaling ensures that the model treats all features fairly and avoids issues related to varying scales, contributing to more stable and accurate results in your machine learning endeavors.

**Random Forest vs Gradient Boosting Decision Trees:**

Random Forest - 
Lets say we have a dataset of 1000 rows. 
We for multiple sets of decision trees (lets say 100 DTs)

**Bootstrapping:**

We sample our data and apply these DTs on the sampled **subset** of the  data. 

How is the sampling done? 
- Row Sampling - We randomly sample rows and apply DT1 for training
With replacement and w/o replacement
Lets say when we are sampling rows - there is a possibility of having duplicaate rows. 

W/o replacement - there is no chance of duplicates. 

- Column/Feature sampling - We randomly sample columns  and apply DT for training
It can be done with replacement or w/o replacement. 

- Combo of row and feature sampling. 

So basically, the data is sampled randomly based on row/feature or both and given to Decision trees training. 

**Aggregation**:
Now once the DTs are trained, next step is prediction.  Now during inference - lets say this is classification problem. 
We give test data to all the DTs. 
All the DTs predict an output. 

In the Aggregation step, we count the outputs from each DTs and decide result based on maximum vote. 

For regression: we decide prediction based on the mean of results from all DTs. 

```Bagging Technique : Bagging is Bootstrapping + Aggregation```

**Bagging ensembles: ** Its a bagging algorithms with different estimators: These estimators can be either logistic regressions/linear Regressions, SVMs, KNNs or DTs. 

**Difference between Bagging ensembles with DT as estimators vs Random Forests**:

In bagging Ensembles the data is sampled randomly on tree level before applying the estimators. For example, when the data is sampled on columns - Its predecided on which columns to be sampled on. 

Where as in Random forests, the randomness is node level: that is when the columns are sampled on node level, the columns are selected randomly whe the nodes in the trees are formed. 

**Why does Random Forest Algorithm work efficiently:** 

If a data is noisy - with high variance and multiple outliers, its very easy for a model to get biased during training. In random forests, because the data is sampled and small subsets are given to multiple DTs, the noie or outliers are divided into multiple subsets lowering the variance in the data. Thus the bias is lowered and DTs rae trained efficinently. 

Random Forest improves the trde offf between Low Bias and Low variance.  
This algorithm gives Low bias and low variance. 

** AdaBoost**:

Adaboost was the first boosting algorithm which was used for complex problems like object detection before deep learning was discovered. 

Boosting is basically ensembles of week learners. These Weak learners are ml algoritms having accuracy just over 50%. Adaboost can use any week learners (linear regression, KNN, SVMs, NN, etc). But by default they use Decision Stumps. Decision stumps basically splits/classify data only in one direction. 

Ada boost is a stage vise additive method. They add the weak learners stage by stage. 
[Refer](https://www.youtube.com/watch?v=sFKnP0iP0K0&list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH&index=98)


Each wea learners is assigned a coffecient/weight based on their accurate predictions. Every time a learner provides accurate results, their importance is improved. 

[<img align="center" src="/Machine learning algorithms/assets/adaboost.PNG" width="600"/>](/Machine learning algorithms/assets/adaboost.PNG)


** Bagging vs Boosting**:

### Type of Model Used: 

We always want models with low bias, low variance. To acheive this, Bagging or boosting models are used. 

In Bagging, we usually take base estimators with Low Bias and high variance. Example, fully grown Decision Trees has  Low Bias and high variance. high accuracy with training data but low accuracy in test data. 

In boosting we consider shallow decision tree (decsion stump) which has Low Bias and High variance. 

### Sequential Vs Parallel

In Bagging, training is done parralelly. Data is divided into subsets and given to Bagging algorithm for parallel training. 

In Boosting : Training is done sequentially. First weak learner is trained and then based on its inaacuracy next kearner is trained and so on. The idea is to imrpove accuracies and decide on the improtance / coefficient for each learner. 

### Weightage: 

In bagging, Each estimators have equal weights. We calculate mean of predictions in regression and in classifier, voting or max output is considered. 

In Boosting, each learner is given an informtance coefficient based on its accuracy. 

**Gradient Boostng**: 

Gradient Boosting build trees one at a time, where each new tree helps to correct errors made by previously trained tree.
The main idea behind this algorithm is to build models sequentially and these subsequent models try to reduce the errors of the previous model. But how do we do that? How do we reduce the error? This is done by building a new model on the errors or residuals of the previous model.

When the target column is continuous, we use Gradient Boosting Regressor whereas when it is a classification problem, we use Gradient Boosting Classifier. The only difference between the two is the “Loss function”. The objective here is to minimize this loss function by adding weak learners using gradient descent. Since it is based on loss function hence for regression problems, we’ll have different loss functions like Mean squared error (MSE) and for classification, we will have different for e.g log-likelihood.


Refer [link](https://www.analyticsvidhya.com/blog/2021/09/gradient-boosting-algorithm-a-complete-guide-for-beginners/) for detailed explanation on gradient boosting. 












