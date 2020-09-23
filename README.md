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

The numerical features we have in our dataset have different ranges. Therefore, the next step 
