----------
**Case study:	Early detection of Type 2
diabetes**
- Framing	as	supervised learning problem
- Evaluating risk stratification algorithms

=====================================


----------

Diabetes Risk Prediction app
* Decision Tree Model
* Trained on over 115,000 survey responses from the CDC's Behavioral Risk Factor Surveillance System Survey
* Data Prep and Model Building @ [Training Code Notebook](https://www.kaggle.com/code/mauiri0192/diabetes-risk-detection).

**Tools:** [Training Code Notebook](https://www.kaggle.com/code/mauiri0192/diabetes-risk-detection), [streamlit](https://streamlit.io), [pandas](https://pandas.pydata.org), [scikit-learn](https://scikit-learn.org/stable/)


**Risk Stratification in Healthcare**

Risk stratification is a fundamental concept in healthcare that involves the systematic categorization of patients into different risk groups based on various factors. This process is essential for optimizing patient care, improving healthcare outcomes, and managing costs effectively. In this article, we'll delve into the key aspects of risk stratification, its significance, and how it has evolved over time. At its core, risk stratification is a way of classifying patients into distinct categories, such as high risk, low risk, or somewhere in between. The primary objective is to predict how patients might fare in terms of their health and well-being and then take specific actions based on these predictions. It's like sorting patients into groups to provide tailored care, interventions, and preventive measures.

A crucial point to note is that risk stratification differs from the process of diagnosis. In diagnosis, healthcare professionals use stringent criteria to identify a specific condition or disease accurately. The consequences of a misdiagnosis can be severe, leading to patients receiving treatments they don't need or facing delayed treatment for critical conditions. Risk stratification, on the other hand, is a bit more flexible. It's not about pinpointing a precise diagnosis but rather assessing the likelihood of different outcomes based on available data. In risk stratification, healthcare providers gather and analyze a wide range of data beyond just medical records. This data might include patient demographics, socioeconomic information, lifestyle factors, and more. These additional factors can significantly influence a patient's risk profile. By considering this comprehensive set of data, healthcare professionals can make more informed decisions about patient care.


In today's healthcare perspective, cost-effectiveness is a significant concern. Risk stratification plays a vital role in addressing this concern by helping healthcare systems identify patients who are at risk of costly complications or hospital readmissions. By targeting interventions towards these high-risk patients, healthcare providers can potentially save resources and reduce overall healthcare costs.

**Real-World Applications**

Risk stratification has real-world applications in healthcare. For example:

1. *Premature Infant Care*: It helps predict the risk of severe morbidity in premature infants, allowing healthcare providers to offer specialized care to those who need it most.

2. *Heart-Related Conditions*: In cases where patients present with heart-related conditions, risk stratification assists in determining whether they should be admitted to a coronary care unit or managed outside the hospital, optimizing resource allocation.

3. *Reducing Hospital Readmissions*: Risk stratification is crucial in predicting the likelihood of patients being readmitted to the hospital after discharge. This information guides healthcare systems in tailoring discharge plans to reduce readmission rates.

**Traditional vs. Modern Approaches**

Traditionally, risk stratification relied on manual scoring systems, which were often time-consuming and limited in scope. However, modern approaches have shifted towards machine learning-based methods. These advanced techniques can handle complex data, provide higher accuracy, and integrate seamlessly into clinical workflows. They are quicker to develop and implement, making them increasingly attractive to healthcare providers.



While machine learning-based risk stratification offers numerous benefits, such as improved accuracy and efficiency, it also poses challenges. The commercialization of these models, ethical concerns related to data privacy and bias, and ensuring that these models fit within established healthcare workflows are areas of active consideration and development.In summary, risk stratification is a vital tool in healthcare that helps healthcare providers categorize patients into risk groups for better-targeted care and interventions. Its evolution from manual scoring systems to modern machine learning-based methods promises to revolutionize patient care, enhance healthcare outcomes, and reduce costs. As technology continues to advance, risk stratification will remain at the forefront of efforts to deliver more efficient and effective healthcare solutions to patients worldwide.



Early detection of type 2 diabetes is a critical healthcare challenge, particularly considering that an estimated 25% of patients in the United States have undiagnosed type 2 diabetes, with similar statistics seen worldwide. Detecting patients with diabetes or those at risk of developing it is essential for implementing interventions to prevent disease progression. This explanation will delve into the importance of this problem and the potential solution involving risk stratification. Identifying individuals with undiagnosed type 2 diabetes or those at high risk of developing it is crucial for several reasons. Firstly, early detection allows for timely interventions that can prevent or slow down the progression of the disease. These interventions might include weight loss programs, dietary changes, and the initiation of medications like Metformin.

Traditionally, healthcare providers used scoring systems similar to the Apgar score to assess the risk of developing type 2 diabetes. These systems considered factors like age, body mass index (BMI), diet, and medication history to assign a risk score. For example, a score below 7 indicated a 1 in 100 risk of developing diabetes, while a score above 20 signaled a high risk with a 1 in 2 chance of developing the disease within ten years. However, these scoring systems have not been as effective as hoped because they were not consistently used. To address the limitations of traditional methods, healthcare researchers are exploring modern approaches, particularly machine learning-based risk stratification. Instead of relying solely on manual scoring systems, these advanced techniques leverage data, including health insurance records, to automatically identify individuals at risk of diabetes or undiagnosed cases.

**Utilizing Data for Risk Stratification**

Modern risk stratification utilizes a broad range of data, such as **patient demographics, medical claims, pharmacy records, and laboratory test results**. While some data might be incomplete or missing, machine learning algorithms can uncover surrogate markers within the available data to predict diabetes risk effectively.
Data censoring, where certain patient data is incomplete or unavailable, presents a challenge. Left censoring is addressed by using available historical data, even if it's limited, to make predictions. Right censoring, where the future data is unknown, is a more complex problem and may require specific techniques to handle.

**Machine Learning Approach**

**L1 Regularized Logistic Regression:** In this context L1 regularized logistic regression is employed as the machine learning algorithm of choice. L1 regularization combines logistic regression with a regularization term that encourages sparsity in the model. This approach allows for the utilization of high-dimensional feature sets while also performing feature selection.Here machine learning approach is posed as an optimization problem, where a loss function is minimized along with a regularization term. In this case, the weights of a linear model are being learned.

The difference between L1 and L2 regularization is where L2 regularization, commonly used in support vector machines, penalizes the L2 norm of the weight vector. In contrast, L1 regularization penalizes the sum of the absolute values of the weights. L1 regularization encourages the model to have zero weights for some features, leading to a sparser solution. Two main benefits of sparsity in models are:
   - **Preventing Overfitting:** In cases where a risk model can effectively use only a small subset of features, sparsity helps prevent overfitting by focusing on the most informative ones.
   - **Interpretability:** Sparse models are more interpretable, as they rely on a limited number of features. This interpretability is crucial for understanding and translating model predictions.

It is also benefetial to create the feature space. It is designed to account for missing data by considering whether certain features were ever observed for patients and handle missing data. For specialists seen by patients, medications taken, and laboratory test results, binary indicators are used. These indicators include whether a feature was ever administered, whether it was low, high, normal, or if the value increased or decreased. To account for the timing of events, the features are computed for different time buckets, such as the last 6 months, last 24 months, and all historical data.

**Evaluation Metrics - Positive Predictive Value (PPV):** The speaker briefly touches upon the evaluation metric used, which is positive predictive value (PPV). PPV measures the fraction of predicted high-risk individuals who actually develop type 2 diabetes. Different levels of predictions are evaluated, allowing for tailored interventions based on risk and cost.

10. **Cost-Effective Interventions:** Examples of interventions, such as sending text messages to high-risk individuals, are discussed. These interventions are cost-effective and can be targeted based on the level of predicted risk.

11. **Comparison with Traditional Approaches:** The lecture concludes by comparing the performance of machine learning-based models with traditional methods. The machine learning model outperforms traditional approaches in identifying high-risk patients.

In summary, this part of the lecture provides insights into the choice of machine learning algorithm, the benefits of L1 regularization for sparsity and interpretability, and the considerations for constructing the feature space and evaluating risk stratification models in the context of predicting type 2 diabetes.


This notebook is using the dataset [Behavioral Risk Factor Surveillance System](https://www.kaggle.com/datasets/cdc/behavioral-risk-factor-surveillance-system). This dataset originally has 330 features (columns), but based on diabetes disease research and above mentioned parameters regarding factors influencing diabetes disease and other chronic health conditions, only select features are included in this analysis.

Important Risk Factors
Research in the field has identified the following as important risk factors for diabetes and other chronic illnesses like heart disease (not in strict order of importance):

- blood pressure (high)
- cholesterol (high)
- smoking
- diabetes
- obesity
- age
- sex
- race
- diet
- exercise
- alcohol consumption
- BMI
- Household Income
- Marital Status
- Sleep
- Time since last checkup
- Education
- Health care coverage
- Mental Health

The **selected features** from the BRFSS 2015 dataset are:

**Response Variable / Dependent Variable:**
*   (Ever told) you have diabetes (If "Yes" and respondent is female, ask "Was this only when you were pregnant?". If Respondent says pre-diabetes or borderline diabetes, use response code 4.) --> DIABETE3

**Independent Variables:**

**High Blood Pressure**
*   Adults who have been told they have high blood pressure by a doctor, nurse, or other health professional --> _RFHYPE5

**High Cholesterol**
*   Have you EVER been told by a doctor, nurse or other health professional that your blood cholesterol is high? --> TOLDHI2
*   Cholesterol check within past five years --> _CHOLCHK

**BMI**
*   Body Mass Index (BMI) --> _BMI5

**Smoking**
*   Have you smoked at least 100 cigarettes in your entire life? [Note: 5 packs = 100 cigarettes] --> SMOKE100

**Other Chronic Health Conditions**
*   (Ever told) you had a stroke. --> CVDSTRK3
*   Respondents that have ever reported having coronary heart disease (CHD) or myocardial infarction (MI) --> _MICHD

**Physical Activity**
*   Adults who reported doing physical activity or exercise during the past 30 days other than their regular job --> _TOTINDA

**Diet**
*   Consume Fruit 1 or more times per day --> _FRTLT1
*   Consume Vegetables 1 or more times per day --> _VEGLT1

**Alcohol Consumption**
*   Heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week) --> _RFDRHV5

**Health Care**
*   Do you have any kind of health care coverage, including health insurance, prepaid plans such as HMOs, or government plans such as Medicare, or Indian Health Service?  --> HLTHPLN1
*   Was there a time in the past 12 months when you needed to see a doctor but could not because of cost? --> MEDCOST

**Health General and Mental Health**
*   Would you say that in general your health is: --> GENHLTH
*   Now thinking about your mental health, which includes stress, depression, and problems with emotions, for how many days during the past 30 days was your mental health not good? --> MENTHLTH
*   Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good? --> PHYSHLTH
*   Do you have serious difficulty walking or climbing stairs? --> DIFFWALK

**Demographics**
*   Indicate sex of respondent. --> SEX
*   Fourteen-level age category --> _AGEG5YR
*   What is the highest grade or year of school you completed? --> EDUCA
*   Is your annual household income from all sources: (If respondent refuses at any income level, code "Refused.") --> INCOME2

**Health care Access**
*   About how long has it been since patient last visited a doctor for a routine checkup? ---> CHECKUP1
