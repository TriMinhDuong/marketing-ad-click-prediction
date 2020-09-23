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

## 2. Data Exploratory Analysis

### 2.1 Exploring Dataset

After loading dataset to dataframe, we started examining dataset to confirm some questions:
1. Does the dataset has any NULL values?
2. Does the dataset has any duplicated instances?

The dataset has 4 instances that did not have values on our target variable "Clicked on Ad". Those instances will be removed. Besides, there are 7 duplicated records. They will also be removed as they might cause bias when building predictive model.

### 2.2 Data Cleaning

#### Is our dataset balanced on 'Clicked on Ad'?

After clearing out the missing values and duplicated records, we see that our dataset is the quite balanced one on the number of potential customers who clicked and did not click on the ad. There are 50.2% of audiences clicking on our ad and 49.8% of audience who did not click on ours.\
We will review this question again before training and building model.

#### Does data has any outliers?

We find that our numerical variables do not seem to be skewed as the mean and the median are roughly similar. With this, it is not neccessary to transform our data.

![Descriptive Statistics](https://github.com/TriMinhDuong/marketing-ad-click-prediction/blob/master/images/numerical_variables-descriptive_statistics.png)

We can see that the range of values on Area Income is quite broad. Additionally, the range of **Age** feature is quite fishy. The minimum and maximum values are not reasonable. We can take a clearer look with the chart below which is showing all current possible values of this feature

![Range of Age Feature](https://github.com/TriMinhDuong/marketing-ad-click-prediction/blob/master/images/age_range.png)

We set our normal range is from 0 years old to 100 years old. We can see that there are 3 values that are out of the normal range of age and 1 negative value which is unreasonable for the Age feature. They are the outliers which should be removed from the dataset.
