# Who do you want to show your marketing ad to maximum your profits?

## 1. Define Problem

Our current problem is to predict who will most likely click on the ad and guide the marketing team to target the appropriate target customers to maximize profitability. Let's consider that I am working for a marketing agency and we utilize data analysis and machine learning to execute more effective marketing campaign.

Firstly, we need to understand how profit and loss is manipulated for digital marketing advertisement. Let's assume that our marketing campaign spend __1000CAD per potential customer__. For each customer that we target with our ad campaign and that clicks on the ad, we will gain __profit of 100CAD__. If we target a customer that do not click on the ad, __a loss of 1050CAD__ will be counted. There will be a situation that the customer that we do not target for the campaign clicks on the ad. For this case, we will gain the __profit of 1100CAD__. This situation will not likely happen for start-up business as we are on the initial stage of building brand awareness.

#### Supplied Data

advertising.csv (source: [Kaggle](https://www.kaggle.com/fayomi/advertising)): data of past audiences who have viewed the marketing ad and whether they clicked on the ad.

Field | Definition | Data Types
----- | ---------- | ----------
Daily Time Spent on Site | How long were the audiences staying on the company website? | Numeric
Age | How old are they? | Numeric
Area Income | What is the average income of the area they were living? | Numeric
Daily Internet Usage | How long were they using the Internet? | Numeric
Ad Topic Line | What was our advertisement about? | Text
City | Which city were they living? | Text
Male | Were they identified as male? | Numeric
Country | Which country were they living? | Text
Timestamp | What time did they see the ad? | DateTime
Clicked on Ad | Did they click the ad? | Binary (1 - Clicked, 0 - Not Clicked)

## 2. Discover Data

### 2.1 Exploring Dataset

After loading dataset to dataframe, we started examining dataset to confirm some questions:
1. Does the dataset has any NULL values?
2. Does the dataset has any duplicated instances?

The dataset has 4 instances that did not have values on our target variable "Clicked on Ad". Those instances will be removed. Besides, there are 7 duplicated records. They will also be removed as they might cause bias when building predictive model.

#### Is our dataset balanced on 'Clicked on Ad'?
After clearing out the missing values and duplicated records, we see that our dataset is the quite balanced one on the number of potential customers who clicked and did not click on the ad. There are 50.2% of audiences clicking on our ad and 49.8% of audience who did not click on ours.
We will review this question again before training and building model.

#### Does data has any outliers?
We find that our numerical variables do not seem to be skewed as the mean and the median are roughly similar. With this, it is not neccessary to transform our data.

![Descriptive Statistics](https://github.com/TriMinhDuong/marketing-ad-click-prediction/blob/master/images/numerical_variables-descriptive_statistics.png)

We can see that the range of values on Area Income is quite broad. Additionally, the range of **Age** feature is quite fishy. The minimum and maximum values are not reasonable. We can take a clearer look with the chart below which is showing all current possible values of this feature.

![Range of Age Feature](https://github.com/TriMinhDuong/marketing-ad-click-prediction/blob/master/images/age_range.png)

We set our normal range is from 0 years old to 100 years old. We can see that there are 3 values that are out of the normal range of age and 1 negative value which is unreasonable for the Age feature. They are the outliers which should be removed from the dataset.

#### Review the features 'Daily Time Spent on Site' and 'Daily Internet Usage'
The feature 'Daily Time Spent on Site' should be equal or less than 'Daily Internet Usage'. There are 3 instances in the dataset which have 'Daily Internet Usage' less than 'Daily Time Spent on Site'. They will be removed from the dataset.

![Daily Internet Usage less than Daily Time Spent on Site](https://github.com/TriMinhDuong/marketing-ad-click-prediction/blob/master/images/delta_less_than_zero.png)

### 2.2 Exploratory Data Analysis

#### Descriptive Statistics for Numerical Features
Let's take a look at the distribution of our numerical features and the correlations between them.

![distplot numerical variables](https://github.com/TriMinhDuong/marketing-ad-click-prediction/blob/master/images/distplot-numerical_variables.png)

The distribution plots for 'Daily Time Spent on Site' and 'Daily Internet Usage' seem to be a bimodal distributions. It means that there could be more than 1 group which is classified by these 2 features. The distribution plots for 'Age' and 'Area Income' are little skewed which could bias our predictive model later. We will use log transformation on these variables to help normalizing the data.

![boxplot numerical variables](https://github.com/TriMinhDuong/marketing-ad-click-prediction/blob/master/images/boxplot-numerical_variables.png)

The boxplots on 'Daily Tim Spent on Site' and 'Daily Internet Usage' show a significant difference between 2 groups, people who clicked on the Ad and people who did not. Furthermore, we will observe a potential trend between these 2 groups with 'Age' and 'Area Income' respectively.

![correlation numerical variables](https://github.com/TriMinhDuong/marketing-ad-click-prediction/blob/master/images/correlation-numerical_variables.png)

The cross-correlation above revealed potential relationship between our target feature 'Clicked on Ad' with other features except gender. Besides, there is no significant relationship between our independent numerical features. Therefore, there could not be collinearity happening between these features.

#### Distribution Plots with Respect to Our Target Feature
![distplot numerical variables with target feature](https://github.com/TriMinhDuong/marketing-ad-click-prediction/blob/master/images/distplot-numerical_variables-target.png)
![pairplot numerical variables with target feature](https://github.com/TriMinhDuong/marketing-ad-click-prediction/blob/master/images/pairplot-numerical_variables-target.png)

We observe the significant differences on 'Daily Time Spent on Site' and 'Daily Internet Usage' between the people who clicked the ad and who did not. The people who clicked ad tended to have less time spent on site and less daily internet usage. In term of Age, the distribution of people who clicked on ad spreaded out from around 20 to 60 years old, while the distribution of people who did not clicked on ad are in the range between 20 and less than 50 years old.

#### Distribution of Categorical Features

There are 4 features in the dataset that are in text format. Due to the time being, the feature 'Ad Topic Line' and 'Timestamp' will be skipped at this moment. The feature 'Ad Topic Line' need to analyzed using Natural Language Processing and the feature 'Timestamp' need to view as the time series data.

Looking at the feature 'City', there are 969 different cities in the dataset out of 1000 instances. It means that there are almost no chance to have many instances from the same city. Therefore, we can confirm that this feature has no predictive power. However, we will have less diversity when it comes down to different countries, so we will take a look at the distribution of our dataset with regard to the feature 'Country'.

![Distribution of Countries](https://github.com/TriMinhDuong/marketing-ad-click-prediction/blob/master/images/distribution_countries.png)

The statistical results show us that there are 237 unique countries in the dataset. On average, each country should have about 4 users and the maximum number of users which were from the same country is 9. This made the feature have very little power to predict whether a user would click on the ad.

## 3. Develop Predictive Model

### 3.1 Prepare Data and Features Engineering

Since we currently do not have data with unknown results for target feature to test our model, we will split this dataset into training and testing sets to train and evaluate model performance. 80% of the dataset will be selected randomly for train set but the rest is for test set.

Before performing machine learning, we will transform some features having skewed distribution to avoid some bias. Those features can make our models underperformed. We will apply logarithmic transformation to reduce the effect of outliers and reduce the distribution's range. From the section 2, we observed that'Age' was right skewed and 'Area Income' was left skewed, so the transformation will be applied to these feature.

As mentioned in section 2, we will not consider the feature 'Ad Topic Line' and 'Timestamp' at this moment. We will not be considering the feature 'City' because it does not have predictive power. Hence, those 3 features will be removed from our train and test sets.

The numerical features we have in our dataset have different ranges. The range of the 'Age' feature is from 0 to 100 while the range of 'Area Income' is from 0 to around 80000. When building the model, the feature 'Area Income' will essentially influence the result more. To avoid this situation, we will normalize our numerical data using min-max scaling approach.

In our model training, we will include the feature 'Country' which is a categorical variable. Hence, we need to transform the data to numerics using LabelEncoder. Since we will only fit the train data to the algorithm, we might experience the situation that some country names in test data did not exist in train data. To avoid our model scrashed when running in production, we will create a value called 'Others' with which those country names will be replaced.

Our train dataset will be looked like below after completing feature engineering process.

![final train set](https://github.com/TriMinhDuong/marketing-ad-click-prediction/blob/master/images/train_set.png)

### 3.2 Model Selection

At this moment, we will only develop the predictive model using either **Logistic Regression** or **Random Forest Classifier**. There are 4 possible outcomes on the evaluation results for each prediction:
- True Positive
- False Positive
- False Negative
- True Negative

As mentioned at the beginning, for each customer that we target for our ad campaign and that clicks on the ad, we will gain profit of 100CAD. If we target a customer that do not click on the ad, there will be a lost of 1050CAD. For each customer that we don't target for our campaign but end up clicking on the ad, the company will gain a profit of 1100CAD. 

Therefore, we will need to lower the number of customers who we target but do not click on the ad. With that, we will need to lower the false positive. Hence, we will use precision score as our evaluation score for model selection. The higher precision score the model has, the better the model is. We will use GridSearchCV to look for the best estimator for each algorithm. Then we will compare the precision scores from 2 models built from Logistic Regression and Random Forest Classifier to decide the model which will be used to predict our test data.

#### Logistic Regression

With logistic regression, we are looking into optimize the value for parameters 'C' with l2 regularization.

The first time, we tried C with value of 0.001, 0.01, 0.1, 1, 10, 100, and 1000. The search algorithm highlighted a value of 1 for C parameter used in the L2-Regularization with the precision score of 98.7066%

![Logistic Regression first try](https://github.com/TriMinhDuong/marketing-ad-click-prediction/blob/master/images/logistic_regression1.png)

We tried the second time to search the values around 1 including 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, and 1.5. The searching gave us a value of 1.2 for C parameter with the precision score of 98.7122%.

![Logistic Regression second try](https://github.com/TriMinhDuong/marketing-ad-click-prediction/blob/master/images/logistic_regression2.png)

New we will look into what the best model that Random Forest Classifier can give us.

#### Random Forest Classifier

We will tune each paramter, including max_depth, n_estimators, max_features, and min_samples_leaf, manually to review the changes of out-of-bag score and precision score. Besides, this will give us some estimations of what range of values we can use for each parameter when start search all toegther.

After reviewing the movements of out-of-bag score and precision score with respect to the changes on each parameter, we look into to find the best combination of the values for the parameters.

The ranges of searching are outlined below
- "n_estimators": [20, 40, 60, 80, 100]
- "max_depth": [7, 8, 9, 10, 11, 12, 13, 14]
- "min_samples_leaf": [2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
- "max_features": ["auto", None, "log2"]

The GridSearchCV gave us the optimized model as below:

![Random Forest Classifier](https://github.com/TriMinhDuong/marketing-ad-click-prediction/blob/master/images/rfc.png)

The best estimators we got from Random Forest Classifier are:
- n_estimators = 20
- max_depth = 8
- min_sample_leaf = 3
- max_features = "auto" (which is the squared root of number of features in train dataset)

This gave us the evaluation score as 96.453%.

#### Best Model

The highest precision score from Logistic Regression was 98.7122% while the highest precision score from Random Forest Classifier was 96.453%. Therefore, the best model which will be used for prediction is generated from Logistic Regression.

![Best Model](https://github.com/TriMinhDuong/marketing-ad-click-prediction/blob/master/images/best_model.png)

After fitting this model to our train dataset, we got a list of coefficent values for all of our independent variables which were used in building model.
- Daily Internet Usage: -5.511
- Daily Time Spent on Site: -4.791
- Age: 3.794
- Area Income: -2.918
- Male: -0.261
- Country: 0.003

We can see that 'Daily Internet Usage', 'Daily Time Spent on Site', 'Age' and 'Area Income' have more impacts to the predictions comparing with the other 2 features.

#### Confusion Matrix

Let's look deeper each possible cases outlined at the beginning and see how we can optimize the model.

![Confusion Matrix - Train 1](https://github.com/TriMinhDuong/marketing-ad-click-prediction/blob/master/images/confusion_matrix-train1.png)

According to the matrix, there are 373 instances which were predicted as '1' (click on the ad) and they actually clicked on the ad. However, there are 5 instances which were predicted as '1' (click on the ad) but these potential customers did not click on the ad in realistic. This incorrect predictions could cause the company in lossing 5,250CAD according to our estimation of costs and profits on running marketing ad campaign.

Let's look at the probability distribution of our model to see if increasing the threshold could be beneficial to reduce the number of false positive, consequently reducing our loss.

![Histogram of Predicted probabilities](https://github.com/TriMinhDuong/marketing-ad-click-prediction/blob/master/images/histogram-train1.png)

Based on the chart, it seems that our model does not carry a lot of ambiguity when it was predicting if someone would click on the ad. We could still increase our threshold to 0.65 to decrease the number of false positive cases, even though our current model seems quite categorical.

Let's apply this new threshold to see how our profits are optimized.

![Confusion Matrix - Train 2](https://github.com/TriMinhDuong/marketing-ad-click-prediction/blob/master/images/confusion_matrix-train2.png)

By increasing our threshold, we are able to decrease the lost for the company by sending the ad to the wrong audience. The revised profits are now 80,850CAD comparing with the previous one as 60,650CAD.

Let's predict the test data with the best model we have selected with default and new threshold.

The summary for the predictions with default threshold as 0.5 is below.

![Confusion Matrix - Test 1](https://github.com/TriMinhDuong/marketing-ad-click-prediction/blob/master/images/confusion_matrix-test1.png)

And then below is the summary for the predictions with threshold as 0.65.

![Confusion Matrix - Test 2](https://github.com/TriMinhDuong/marketing-ad-click-prediction/blob/master/images/confusion_matrix-test2.png)

Our test set has a sample size of 200 customers. **With the new threshold, our predicted overall profit would be 17,100CAD**. This includes:
- Profit of 9,400CAD from true positives
- Profit of 7,700CAD from false negatives
- Loss of 0CAD from false positives

Indeed, the false negatives are extremely rewarding considering this particular problem. There could be a lost from the false positives but the count would be less than or equal to 2 which is 2 out of 200 (1%). We can state that the results are excellent considering the margin of error and the predicted profit.

## 4. Actionable Recommendations

According to our model and analysis, we can identify the potential customers by getting information of:

- Daily Time Spent on Site
- Daily Internet Usage
- Area Income
- Age
By getting this information, we can target new customers with our ad campaign to maximize the chance of a return on investment.

From our exploratory data analysis, our targeted population would be customers with:

- Lower income
- Less time spent on the internet
- Less tiem spent on site
- Older than our average sample (mean around 35 years old)
Also by increasing our threshold from our model, we can minimize the number of wrong targets to minize loss which will optimize our business approach.
