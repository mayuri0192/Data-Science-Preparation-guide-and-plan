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










