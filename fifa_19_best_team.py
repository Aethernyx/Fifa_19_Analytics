#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


# In[2]:


# Changing the settings for our pandas dataframes to display up to 100 columns, and 100 rows

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)


# In[3]:


# Loading the dataset to a variable called data
filename = "/Users/alex/desktop/python_work/data-sets/fifa_data.csv"
data = pd.read_csv(filename)


# In[ ]:





# ## The goal of this project is to use Machine Learning Techniques to build the fastest, best overall starting U_21 team

# To preface I want to build a team predicated on speed of the best players that are 21 years old or younger.
# For my team I want to choose the best players in the following positions:
# 1 ST, 1 LW, 1 RW, 1 LM, 1 RM, 1 CDM, 1 RB, 1 LB, 2 CB, 1 GK

# In[4]:


# Creating a new dataframe named u_21 using a copy of the data dataframe

u_21 = data.copy()
u_21 = u_21[u_21['Age'] <= 21]


# In[5]:


u_21.head()


# In[6]:


# Off the bat I see from looking at the dataframe that there are a bunch of categories that are not relevant
# to us, so I will remove them

u_21.drop(['Unnamed: 0', 'Photo', 'Flag', 'Club', 'Club Logo', 'Special', 'International Reputation', 
        'Body Type', 'Real Face', 'Joined', 'Loaned From', 'Contract Valid Until', 'Release Clause'], 
        axis=1, inplace=True)


# In[7]:


u_21.info()


# In[8]:


# Let's see what the percentage of null values is for each feature

100 * u_21.isnull().sum() / len(u_21)


# In[9]:


# Checking to see which players have NaN value for the 'LS' feature

u_21[u_21['LS'].isnull()]


# In[10]:


# It looks like a majority of the features with null values have less than 1% of null values in total.
# The remaining features have 11% of their values as null values. Upon further investigation it looks like
# the NaN values are for GoalKeepers, I will create a new dataframe just for GoalKeepers and remove them from this
# data set.

u_21_gk = u_21.copy()
u_21_gk = u_21_gk[u_21_gk['Position'] == 'GK']

u_21.dropna(axis=0, inplace=True)


# In[11]:


# I want to create some dummy variables for some features that I would like to feed to a model that I will build
work_rate_dummies = data['Work Rate'].map({'Low/Low': 1, 'Low/ Medium': 2, 'Low/ High': 3, 'Medium/ Low': 4,
                                      'Medium/ Medium': 5, 'Medium/ High': 6, 'High/ Low': 7, 'High/ Medium': 8,
                                      'High/ High': 9})

preferred_foot = pd.get_dummies(u_21['Preferred Foot'], drop_first=True)

u_21['Work Rate'] = work_rate_dummies
u_21 = pd.concat([u_21.drop('Preferred Foot', axis=1), preferred_foot], axis=1)


# In[12]:


# Renaming the Column 'Right' to 'right_footed'
# 1 is yes, and 0 is no

u_21.rename(columns={'Right': 'right_footed'}, inplace=True)


# In[13]:


# I want to remove lbs from the Weight column so I can conver the feature to a numeric data type

u_21['Weight'] = u_21['Weight'].apply(lambda x: x[:3])
u_21['Weight'] = pd.to_numeric(u_21['Weight'])


# In[14]:


# I also want to convert the Height feature to a numeric value. It may be unconventional but, I want to remove the ' 
# apostrophe and replace it with a decimal. So 5'7 would become 5.7

u_21['Height'] = u_21['Height'].apply(lambda x: x.replace("'", '.'))
u_21['Height'] = pd.to_numeric(u_21['Height'])


# In[15]:


# I want to convert all the positional overall columns we need for our team to numerical values. 
# I am going to write a function that would convert them to numbers. An example is converting 85+3 to 88, 
# and make it an int value.

def convert_to_number(num):
    num_1 = num.split('+')[0]
    num_2 = num.split('+')[1]
    return int(num_1) + int(num_2)


# In[16]:


# Converting the columns to numbers

u_21['ST'] = u_21['ST'].apply(convert_to_number)
u_21['LW'] = u_21['LW'].apply(convert_to_number)
u_21['RW'] = u_21['RW'].apply(convert_to_number)
u_21['LAM'] = u_21['LAM'].apply(convert_to_number)
u_21['RAM'] = u_21['RAM'].apply(convert_to_number)
u_21['LM'] = u_21['LM'].apply(convert_to_number)
u_21['LCM'] = u_21['LCM'].apply(convert_to_number)
u_21['RCM'] = u_21['RCM'].apply(convert_to_number)
u_21['RM'] = u_21['RM'].apply(convert_to_number)
u_21['CDM'] = u_21['CDM'].apply(convert_to_number)
u_21['LB'] = u_21['LB'].apply(convert_to_number)
u_21['CB'] = u_21['CB'].apply(convert_to_number)
u_21['RB'] = u_21['RB'].apply(convert_to_number)


# In[17]:


# Removing the additional columns I decided that I don't need

u_21.drop(['LS', 'RS', 'LF', 'CF', 'RF', 'CM', 'CAM', 'LWB', 'LDM', 'RDM', 'RWB', 'RCB', 'LCB', 'GKDiving', 'GKHandling',
          'GKKicking', 'GKPositioning', 'GKReflexes'], axis=1, inplace=True)


# In[18]:


# Since I want my team to be the fastest team possible, I am going to filter out players who have a sprint speed 
# lower than a 70.

u_21 = u_21[u_21['SprintSpeed'] >=70]


# In[19]:


# Checking to see if we have all the positions we want in our updated dataframe

u_21['Position'].unique()


# In[20]:


u_21.head(2)


# In[ ]:





# In[21]:


# Before I get to EDA I want to clean up the u_21_gk dataframe

# Dropping the uneccesary columns
u_21_gk.drop(u_21_gk.columns[16:42], axis=1, inplace=True)
u_21_gk.drop(['Crossing', 'Finishing', 'HeadingAccuracy', 'Volleys', 'Dribbling', 'Curve', 
             'FKAccuracy', 'BallControl', 'ShotPower', 'Stamina', 'LongShots',
             'Aggression', 'Interceptions', 'Positioning', 'Penalties', 'Marking',
             'StandingTackle', 'SlidingTackle'], axis=1, inplace=True)

# Converting the work rate and preferred rate features to dummies
work_rate_dummies = u_21_gk['Work Rate'].map({'Low/Low': 1, 'Low/ Medium': 2, 'Low/ High': 3, 'Medium/ Low': 4,
                                      'Medium/ Medium': 5, 'Medium/ High': 6, 'High/ Low': 7, 'High/ Medium': 8,
                                      'High/ High': 9})

preferred_foot = pd.get_dummies(u_21_gk['Preferred Foot'], drop_first=True)

u_21_gk['Work Rate'] = work_rate_dummies
u_21_gk = pd.concat([u_21_gk.drop('Preferred Foot', axis=1), preferred_foot], axis=1)

# Converting the Weight and Height features
u_21_gk['Weight'] = u_21_gk['Weight'].apply(lambda x: x[:3])
u_21_gk['Weight'] = pd.to_numeric(u_21_gk['Weight'])

u_21_gk['Height'] = u_21_gk['Height'].apply(lambda x: x.replace("'", '.'))
u_21_gk['Height'] = pd.to_numeric(u_21_gk['Height'])

# Changing the name of the right footed column from Right to right_footed
u_21_gk.rename(columns={'Right': 'right_footed'}, inplace=True)


# In[22]:


u_21_gk.head(3)


# In[ ]:





# ## Exploratory Data Analysis

# Things to explore
# - Most frequent jersey Number
# - Overall vs. Sprint Speed
# - Age count plot
# - Weak Foot countplot
# - Work rate countplot
# - Height and Weight Distribution
# - Overall Distribution 
# - Jointplots between similar attributes
# - Countplot of preferred foot
# - Boxplot of Overall with Age as a hue

# In[23]:


sns.set_style('darkgrid')


# In[24]:


# Distribution of Overalls

plt.figure(figsize=(12, 5))
sns.distplot(u_21['Overall'], bins=30, kde=False, color='red')
plt.title('Distribution of Overall Rating')


# In[25]:


# Distribution of Sprint Speeds

plt.figure(figsize=(12, 5))
sns.distplot(u_21['SprintSpeed'], kde=False, color='green', bins=30)
plt.title('Distribution of Sprint Speed')


# In[26]:


# Joinplot comparing the distribution of Overalls with the distribution of Sprint Speeds

sns.jointplot(x='Overall', y='SprintSpeed', data=u_21, ratio=5, height=8, alpha=0.4, color='black')
plt.title('Overall vs. Sprint Speed')
plt.show()


# In[27]:


# Comparing Sprint Speed with Acceleratin to see if there's a relationship between the two

sns.jointplot(x='SprintSpeed', y='Acceleration', data=u_21, ratio=5, height=8, alpha=0.4, color='purple')
plt.title('Sprint Speed vs. Aceeleration')
plt.show()


# In[28]:


u_21.head()


# In[29]:


# Since I see a linear relationship between SprintSpeed and Acceleration I want to plot a 3D plot to see if there's
# also a linear relationship between Agility, SprintSpeed, and Acceleration
from mpl_toolkits.mplot3d import Axes3D
sns.set_style('white')

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(u_21['SprintSpeed'], u_21['Acceleration'], u_21['Agility'], alpha=0.4, s=60)
ax.view_init(30, 185)
ax.set_xlabel('SprintSpeed')
ax.set_ylabel('Acceleration')
ax.set_zlabel('Agility')
ax.set_title('Sprint Speed vs. Acceleration vs. Agility')
plt.show()


# In[30]:


# Looking at a jointplot comparing short passing with long passing

sns.set_style('darkgrid')

sns.jointplot(x='ShortPassing', y='LongPassing', data=u_21, height=8, color='red', alpha=0.4)
plt.title('Short Passing vs. Long Passing')
plt.show()


# In[31]:


# It looks like there's a linear relationship between short passing and long passing as well. I want to see if 
# there's also a linear realtionship between short passing and long passing with Crossing
sns.set_style('white')

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(u_21['ShortPassing'], u_21['LongPassing'], u_21['Crossing'], alpha=0.4, s=60)
ax.set_xlabel('Short Passing')
ax.set_ylabel('Long Passing')
ax.set_zlabel('Crossing')
ax.set_title('Short Passing vs. Long Passing vs. Crossing')
plt.show()


# In[32]:


# I want to plot a jointplot between Dribbling and BallControl
sns.set_style('darkgrid')

sns.jointplot(x='Dribbling', y='BallControl', data=u_21, color='blue', height=8, kind='hex')
plt.title('Dribbling vs. Ball Control')
plt.show()


# In[33]:


# Now I want to use plotly to plot distribution of individual position overall ratings.

import plotly
import cufflinks as cf
cf.go_offline()

u_21['ST'].iplot(kind='hist', xTitle='Striker Overall', yTitle='Count')


# In[34]:


# Plotting distribution of Left and Right Wings

u_21['LW'].iplot(kind='hist', xTitle='Left / Right Wing Overall', yTitle='Count', colorscale='blues')


# In[35]:


# Plotting distribution of Left / Right attacking mid Overalls

u_21['LAM'].iplot(kind='hist', xTitle='Left / Right Attacking Mid Overall', yTitle='Count', colorscale='greens')


# In[36]:


# Plotting distribution of Center Backs

u_21['CB'].iplot(kind='hist', xTitle='Center Backs Overall', yTitle='Count', colorscale='reds')


# In[37]:


# I want to plot a count plot of the ages of under 21 players
sns.set_style('whitegrid')

plt.figure(figsize=(10, 5))
sns.countplot(x='Age', data=u_21, palette='Set2')
plt.title('Age Count')
plt.show()


# In[38]:


# Next I want to plot a countplot of weakfoot distributions

sns.countplot(x='Weak Foot', data=u_21)
plt.title('Weak Foot Count')
plt.show()


# In[39]:


# I also want to see the counts of right footed vs. left footed players. 0 is lefties, 1 is righties

sns.countplot(x='right_footed', data=u_21)
plt.title('Right Footed vs. Left Footed')
plt.show()


# In[40]:


# I also want to see the counts of Work Rate

sns.countplot(x='Work Rate', data=u_21)
plt.title('Work Rate Count')
plt.show()


# In[41]:


# Counting total left footed vs. right footed players, and percentage of righties vs. lefties

print('Total Right Footed Players: {}'.format(len(u_21[u_21['right_footed'] == 1])))
print('Percentage of Right Footed Players: {}'.format(round(100.0 * 
                                                            len(u_21[u_21['right_footed'] == 1]) / len(u_21), 2)))
print('\n')
print('Total Left Footed Players: {}'.format(len(u_21[u_21['right_footed'] == 0])))
print('Percentage of Left Footed Players: {}'.format(round(100.0 * 
                                                     len(u_21[u_21['right_footed'] == 0]) / len(u_21), 2)))


# In[42]:


# I want to plot the average sprint speed per position vs. the average overall per position
# First I will create new series for sprint speed and overall per position

average_position_speed = u_21.groupby('Position')['SprintSpeed'].mean().reset_index()
average_position_overall = u_21.groupby('Position')['Overall'].mean().reset_index()


# In[43]:


plt.figure(figsize=(15, 8))

n=1
t=2
d=25
w=0.8

x = [t*element + w*n for element in range(d)]

plt.bar(x, average_position_overall['Overall'], label='Overall')

# Adding the SprintSpeed averages per position side by side with the Overall bar graph

n=2
t=2
d=25
w=0.8

x = [t*element + w*n for element in range(d)]

ax = plt.subplot()
plt.bar(x, average_position_speed['SprintSpeed'], label='Sprint Speed')
ax.set_xticks(x)
ax.set_xticklabels(average_position_speed['Position'])
ax.set_xlabel('Position')
ax.set_ylabel('Rating')
plt.ylim(50, 90)
plt.legend()

plt.title('Average Positional Overall and Sprint Speed')

plt.show()


# On Average it looks like LAM are our fastest players, while RF are on average the best overall players.
# It also looks like on average RDM are our slowest players, while LWB on average are our worst players.
# 
# However this can be misleading depending on how many players are in each position. To combat this I want to plot a boxplot to see the distributions for each position and their Sprint Speeds / Overall.

# In[44]:


# Plotting box plot for Sprint Speed for each Position
plt.figure(figsize=(15, 8))

sns.boxplot(y='SprintSpeed', x='Position', data=u_21, orient='v')
plt.title('Positional Sprint Speed Distributions')
plt.show()


# In[45]:


# Plotting box plot for Sprint Speed for each Position
plt.figure(figsize=(15, 8))

sns.boxplot(x='Position', y='Overall', data=u_21, orient='v')
plt.title('Positional Overall Distributions')


# In[46]:


# Checking to see how many players are RF, LF, or LAM

print("Total RF: {}".format(len(u_21[u_21['Position'] == 'RF'])))

print("Total LF: {}".format(len(u_21[u_21['Position'] == 'LF'])))

print("Total LAM: {}".format(len(u_21[u_21['Position'] == 'LAM'])))


# In[47]:


# Looking at the distribution of Heights
sns.set_style('darkgrid')

plt.figure(figsize=(10, 5))
sns.distplot(u_21['Height'], bins=10, kde=False, color='red')
plt.title('Distribution of Heights')
plt.show()


# In[48]:


# I'm curious if Height has an effect on sprint speed, and if it has an effect on overall. I will plot both

sns.jointplot(x='Height', y='SprintSpeed', data=u_21, height=8, alpha=0.4)
plt.title('Height vs. Sprint Speed')
plt.show()


# In[49]:


sns.jointplot(x='Height', y='Overall', data=u_21, height=8, alpha=0.4, color='black')
plt.title('Height vs. Overall')
plt.show()


# In[50]:


# Creating a Pairgrid to compare certain features with each other in one plot

plt.figure(figsize=(12, 12))

g = sns.PairGrid(u_21, x_vars=['Overall', 'ST', 'LW', 'RW', 'LM', 'RM', 'CDM', 'LB', 'RB', 'CB'],
                 y_vars=['Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys',
                            'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',
                            'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
                            'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions',
                            'Positioning', 'Vision', 'Composure', 'Marking', 'StandingTackle',
                            'SlidingTackle', 'SlidingTackle'])
g.map(plt.scatter, alpha=0.4)
g


# In[99]:



plt.figure(figsize=(15, 10))
sns.heatmap(u_21.drop(['ID', 'Age', 'Skill Moves',
                       'Jersey Number'], axis=1).corr(), annot=False, cmap='coolwarm')
plt.title('Heatmap of Atrribute correlations')
plt.show()


# For our defenders (CDM, RB, LB, CB) it looks like the following have a very defined linear relationship:
# - Standing Tackle
# - Sliding Tackle
# - Marking
# - Interceptions
# - Strength
# - Agression
# - Stamina
# - Reactions
# - Short Passing
# 
# For our offensive players (ST, LW, RW, LAM, RAM)it looks like the following have a very defined linear relationship:
# - Composure
# - Vision
# - Positioning
# - Long Shots
# - Shot Power
# - Reactions
# - Ball Control
# - Long Passing
# - Curve
# - Dribbling
# - Volleys
# - Short Passing
# - Finishing
# - Crossing

# ## Building a Linear Regression model on the data

# In[51]:


# Importing the modules I will use for the model

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[52]:


# Creating the feature variable X and label variable y for the model

X = u_21[['Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys',
                            'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',
                            'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
                            'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions',
                            'Positioning', 'Vision', 'Composure', 'Marking', 'StandingTackle',
                            'SlidingTackle', 'SlidingTackle', 'Weak Foot', 'right_footed', 'Height', 'Weight',
         'Potential']]
y = u_21['Overall']


# In[53]:


# Splitting the data into train sets and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[54]:


# Scaling and transforming the X_train features

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)


# In[55]:


# Scaling the X_test features

X_test_scaled = scaler.transform(X_test)


# In[56]:


# Creating an object for the model and fitting the model

model = LinearRegression()

model.fit(X_train_scaled, y_train)


# In[57]:


# Checking the r-squared for the model

model.score(X_test_scaled, y_test)


# In[58]:


# I want to test to see how well our model performs. I am going to use mean_squared_error, mean_absolute_error, and
# root_mean_squared_error

# Importing the modules

from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[59]:


# Creating a new predictions variable, and then testing to see how much error there is in the model

predictions = model.predict(X_test_scaled)


# In[60]:


print("Mean Absolute Error: {}".format(mean_absolute_error(y_test, predictions)))
print('\n')
print("Mean Squared Error: {}".format(mean_squared_error(y_test, predictions)))
print('\n')
print("Root Mean Sqaured Error: {}".format(np.sqrt(mean_absolute_error(y_test, predictions))))


# In[61]:


X.head()


# The Linear Regression Model performs very well based off of the mean squared, root mean squared and absolute error values. Since an Overall rating can be anywhere from 1-99, the mean squared error is around 4%, which means on average the predictions are potentially off by 4% 
# 
# - An example to better understand - if the actual overall was 87, the model might potentially predict anywhere from 83-91 based off of the mean squared error which is still a pretty accurate description.

# In[62]:


# Looking at the coefficients and bias for this model in a dataframe

weights = pd.DataFrame(model.coef_, index=X.columns).transpose()
bias = model.intercept_
weights['Bias'] = bias
weights = weights.transpose()
weights.rename(columns={0: 'Weights'}, inplace=True)

weights


# To find our best players for each position I'm going to use the coefficients and bias for each feature and pick the players with the highest score for each position

# In[63]:


# Creating an array with all of our coefficients for each feature in it

coef = model.coef_
coef


# In[64]:


# Multiplying each row in each column by their features proper coefficient

rating = np.multiply(coef, X.values)


# In[65]:


# Summing the multiplied results of features x coefficients per each row

scores = []

for i in range(len(rating)):
    row_rating = rating[i].sum()
    scores.append(row_rating)


# In[66]:


# Adding the bias to each row

scores = [score + model.intercept_ for score in scores]


# In[67]:


# Adding the values to the u_21 dataframe under the column name 'final_score'

u_21['final_score'] = scores


# In[68]:


# Sorting the dataframe based on final score in descending order

u_21.sort_values('final_score', ascending=False, inplace=True)


# In[69]:


u_21.head()


# In[70]:


# Finding our Centerbacks

center_backs = u_21[(u_21['Position'] == 'CB') | (u_21['Position'] == 'RCB') | (u_21['Position'] == 'LCB')]
center_backs = u_21[(u_21['ID'] == 235243) | (u_21['ID'] == 240130)].reset_index(drop=True)


# In[71]:


# Finding the Left and Right Backs

lb = u_21[(u_21['Position'] == 'LB') | (u_21['Position'] == 'LWB')]
lb = u_21[u_21['ID'] == 235212].reset_index(drop=True)


rb = u_21[(u_21['Position'] == 'RB') | (u_21['Position'] == 'RWB')]
rb = u_21[u_21['ID'] == 231281].reset_index(drop=True)


# In[72]:


# Finding the Striker

striker = u_21[(u_21['Position'] == 'ST') | (u_21['Position'] == 'RS') | (u_21['Position'] == 'LS')]

striker = u_21[u_21['ID'] == 230666].reset_index(drop=True)


# In[73]:


# Finding the left and right wings

lw = u_21[(u_21['Position'] == 'LW') | (u_21['Position'] == 'LF')]
lw = u_21[u_21['ID'] == 231677].reset_index(drop=True)

rw = u_21[(u_21['Position'] == 'RW') | (u_21['Position'] == 'RF')]
rw = u_21[u_21['ID'] == 231443].reset_index(drop=True)


# In[74]:


# Finding the midfielders, right, left and central defensive mid

rm = u_21[(u_21['Position'] == 'RM') | (u_21['Position'] == 'RAM')]
rm = u_21[u_21['ID'] == 231747].reset_index(drop=True)

lm = u_21[(u_21['Position'] == 'LM') | (u_21['Position'] == 'LAM')]
lm = u_21[u_21['ID'] == 229906].reset_index(drop=True)

cdm = u_21[(u_21['Position'] == 'CDM') | (u_21['Position'] == 'RDM') | (u_21['Position'] == 'LDM')]
cdm = u_21[u_21['ID'] == 228702].reset_index(drop=True)


# In[ ]:





# In[75]:


# Now that I have my lineup of field players, I need to find my goalkeeper. First I want to look at all
# attributes in relation to a GK Overall

g = sns.PairGrid(u_21_gk, y_vars=['Potential', 'Height', 'Weight', 'ShortPassing', 'LongPassing', 
                                'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'Jumping',
                                'Strength', 'Vision', 'Composure', 'GKDiving', 'GKHandling',
                                'GKKicking', 'GKPositioning', 'GKReflexes'], x_vars=['Overall'], height=3)
g.map(plt.scatter, alpha=0.4)


# In[76]:


# Creating the feature and label variables

X = u_21_gk[['Potential', 'Reactions', 'Composure', 'GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning',
         'GKReflexes']]

y = u_21_gk['Overall']


# In[77]:


# Creating an object for our second model

model_2 = LinearRegression()


# In[78]:


# Creating train and test feature and label variables

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[79]:


# Scaling the train and test features

scaler_2 = StandardScaler()

X_train_scaled = scaler_2.fit_transform(X_train)

X_test_scaled = scaler_2.transform(X_test)


# In[80]:


# Fitting the model with our X_train_scaled features and y_train labels

model_2.fit(X_train_scaled, y_train)


# In[81]:


# Creating our predictions based off of our model

predictions_2 = model_2.predict(X_test_scaled)


# In[82]:


# Testing how accurate our model is

print('Mean Sqaured Error: {}'.format(mean_squared_error(y_test, predictions_2).round(2)))
print('\n')
print('Mean Absolute Error: {}'.format(mean_absolute_error(y_test, predictions_2).round(2)))
print('\n')
print('Root Mean Squared Error: {}'.format(np.sqrt(mean_squared_error(y_test, predictions_2)).round(2)))


# The model is extremely accurate with only a small margin of error

# In[83]:


# Creating an array for the model's coefficients

gk_coef = model_2.coef_


# In[84]:


# Multiplying our coefficients by each feature row

ratings = np.multiply(gk_coef, X.values)


# In[85]:


# Summing up each row and adding them to a new list scores

scores = []

for row in ratings:
    scores.append(row.sum())


# In[86]:


# Adding the bias to each score

scores = [score + model_2.intercept_ for score in scores]


# In[87]:


# Adding the scores to our dataframe and then sorting by final_score in descending order

u_21_gk['final_score'] = scores
u_21_gk.sort_values('final_score', ascending=False, inplace=True)


# In[88]:


gk = u_21_gk[u_21_gk['ID'] == 230621].reset_index(drop=True)


# In[ ]:





# In[89]:


# Creating a dataframe with our starting lineup

final_team = data[(data['ID'] == 235243) | (data['ID'] == 240130) | (data['ID'] == 235212) | 
                 (data['ID'] == 231281) | (data['ID'] == 230666) | (data['ID'] == 231677) | 
                 (data['ID'] == 231443) | (data['ID'] == 231747) | (data['ID'] == 229906) | 
                 (data['ID'] == 228702) | (data['ID'] == 230621)]

final_team.drop(['Unnamed: 0', 'Photo', 'Nationality', 'Flag', 'Club Logo', 'Joined', 'Loaned From'], axis=1,
               inplace=True)


# In[90]:


final_team


# In[ ]:




