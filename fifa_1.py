import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

# Loading the dataset to a variable called data
filename = "/Users/alex/desktop/python_work/data-sets/fifa_data.csv"
data = pd.read_csv(filename)

"""I want to find out what the most commonly used number is in the game, so let's plot a histogram to determine this"""
# Setting the numbers column to a variable named 'jersey_numbers'
jersey_numbers = data['Jersey Number']

#print(jersey_numbers)

#ax = plt.subplot(1, 1, 1)
#plt.hist(jersey_numbers, range=(0, 100), bins=100, edgecolor='black')
#plt.xlabel("Jersey Number")
#plt.ylabel("Total Players")
#plt.xlim(0, 100)
#plt.ylim(0, 750)
#ticks = ax.set_xticks(range(100))
#ax.set_xticklabels(range(0, 100), fontsize=5)
#plt.show()

"""We conclude from the graph that '8' is the most common jersey number. Let's test that out with some code."""

jersey_numbers_count = data.groupby('Jersey Number')['ID'].count()
#print(jersey_numbers_count)

"""The code above groups by jersey numbers, and outputs the total count of players who wear each jersey number.
We see that number '8' is the most common jersey number. We also see that there is only one person wearing the number
'85' which makes 85 the least common number."""


"""I'm curious to see how relevant a player's potential rating is to their current sprint speed."""
# Setting up x, and y variables
data_1500_entries = data[0:1501]
x = data_1500_entries['SprintSpeed']
y = data_1500_entries['Potential']

#print(np.any(np.isnan(x)))
#print(np.all(np.isfinite(x)))

# reshaping x since we're only using one variable
x_matrix = x.values.reshape(-1, 1)

# Fitting the regression
reg = LinearRegression()

reg.fit(x_matrix, y)
weights = reg.coef_
bias = reg.intercept_
from sklearn.feature_selection import f_regression

#print(f_regression(x_matrix, y))
#print(reg.score(x_matrix, y))

"""So from the info above, we see that SprintSpeed is very significant, so it definitely is a factor in determining
player potential. However the R-squared is only 0.05, which says that Sprint Speed only accounts for 5% of 
the variability of Potential."""

"""Let's see how Sprint Speed and Work Rate impact Potential."""
# First we must turn Work Rate into Dummy variables
data_dummies = data['Work Rate'].map({'Low/Low': 1, 'Low/ Medium': 2, 'Low/ High': 3, 'Medium/ Low': 4,
                                      'Medium/ Medium': 5, 'Medium/ High': 6, 'High/ Low': 7, 'High/ Medium': 8,
                                      'High/ High': 9})
speed_and_work_rate = data.copy()
speed_and_work_rate['Work Rate'] = data_dummies

speed_and_work_rate_1500 = speed_and_work_rate[0:1501]

x2 = speed_and_work_rate_1500[['SprintSpeed', 'Work Rate']]
y2 = speed_and_work_rate_1500["Potential"]

reg = LinearRegression()
reg.fit(x2, y2)
#print(reg.coef_)
#print(reg.score(x2, y2))
ss_f_stat = f_regression(x2, y2)[0][0].round(3)
ss_p_value = f_regression(x2, y2)[0][1].round(3)
#print(ss_f_stat)
#print(ss_p_value)

# Let's perform VIF to see if Sprint Speed and Work Rate have multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = speed_and_work_rate_1500[['SprintSpeed', 'Work Rate']]

vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif['Features'] = variables.columns
#print(vif)

"""It turns out that Sprint Speed and Work Rate have extremely high multicollinearity, so we need to test 
other things."""


"""Let's see what the spread is of wage"""

# Let's Start by plotting the wages in a histogram
new_data = data.drop(['Unnamed: 0', 'Photo', 'Flag', 'Club Logo', 'Special', 'International Reputation', 'Real Face',
                      'Joined', 'Loaned From'], axis=1)

wages = new_data['Wage'].apply(lambda x: x.split('K')[0])
wages_2 = wages.apply(lambda x: x.split('€')[-1])

wages_float = wages_2.astype(float)

sorted_wages = sorted(wages_float)


#plt.hist(sorted_wages, bins=200, edgecolor='black')
#plt.xlabel("Weekly Wage")
#plt.ylabel('Total')
#plt.show()

#ax = plt.subplot
#sns.distplot(sorted_wages)
#plt.show()

"""By plotting a histogram, and a distribution we see a right skewed unimodal distribution. However the 
mehtods used above do not give us a good visual depiction of wage spread. Let's try to create a boxplot to see if """
#figure_2 = plt.figure(figsize=(10, 5))
#plt.subplot()
#sns.boxplot(sorted_wages)
#plt.xlabel("Wage Amount")
#plt.show()

#print(np.quantile(sorted_wages, 0.25))
#print(np.median(sorted_wages))
#print(sorted_wages.describe())



"""The box plot shows us something interesting. The interquartile range is 8, which means 50% of our data
falls in this range. There is an extreme amount of outliers to the right of our boxplot. This also reveals
that our data is extremely right-skewed. Based off of this we attain that although there are a lot of outliers
in terms of wage, a heavy majority of players make between 1-9 thousand euros a week. Only a slight amount of players
make in excess of that"""



"""I want to build a team with high overall ratings, and top speed. First I want to create a cluster of 
overall vs. speed"""
from sklearn.cluster import KMeans

data_for_k_cluster = new_data[['Overall', 'SprintSpeed']]
data_for_k_cluster = data_for_k_cluster.dropna(axis=0)
data_for_k_cluster = data_for_k_cluster[:1501]
#q = data_for_k_cluster.quantile(0.75)
#data_for_k_cluster = data_for_k_cluster[data_for_k_cluster[['Overall', 'SprintSpeed']]>q]

#print(data_for_k_cluster.isnull().sum())

x = data_for_k_cluster['SprintSpeed']
y = data_for_k_cluster['Overall']
x_matrix = x.values.reshape(-1, 1)
y_matrix = y.values.reshape(-1, 1)

kmeans = KMeans(4)
kmeans.fit(y_matrix)
cluster_prediction = kmeans.fit_predict(y_matrix)

"""wcss = []

for i in range(1, 8):
    kmeans = KMeans(i)
    kmeans.fit(y_matrix)
    wcss_iter = kmeans.inertia_
    wcss.append(wcss_iter)


plt.plot(range(1, 8), wcss)
plt.show()"""

#plt.scatter(x, y, c=cluster_prediction, cmap='rainbow')
#plt.xlabel("Sprint Speed")
#plt.ylabel("Overall Rating")
#plt.show()


"""I want to find out what a strikers Potential is based on SprintSpeed, Finishing, HeadingAccuracy, Dribbling,
and Acceleration"""

striker = new_data[['SprintSpeed', 'Acceleration', 'Finishing', 'HeadingAccuracy', 'Dribbling', 'Potential']]
striker = striker.dropna(axis=0)

x = striker.drop('Potential', axis=1)
y = striker['Potential']


reg = LinearRegression()
reg.fit(x, y)
r_sqaured = reg.score(x, y)


"""I want to build the best speed U-21 team possible by analyzing specific traits per position, and possibly building a
regression to predict overall based on these traits and potential.

Let's start with a striker"""

u_21 = new_data.copy()
u_21 = u_21[u_21['Age'] <= 21]

u_21_striker = u_21[(u_21['Position'] == 'ST') & (u_21['SprintSpeed'] > 85)]
u_21_striker = u_21_striker.drop(u_21_striker.columns[27:45], axis=1).reset_index(drop=True)

"""I want to visualize the most important measurements for a striker. Here that would be SprintSpeed, Acceleration,
Agility, Finishing, ShotPower, HeadingAccuracy, Volleys, Dribbling, Curve, BallControl, LongShots.

Let's start by creating a barplot to show who has the fastest Sprint Speed."""
x = range(len(u_21_striker))
y = u_21_striker['SprintSpeed']

n = 1
t = 3
d = 28
w = 0.8

x_values1 = [t*element + w*n for element in range(d)]
plt.figure(figsize=(10, 5))
ax = plt.subplot(1, 1, 1)
plt.bar(x_values1, y, label='Sprint Speed')
plt.xlabel("Player ID")
plt.ylabel("Total Speed")
ax.set_xticks(x_values1)
ax.set_xticklabels(u_21_striker['ID'], rotation=90)
plt.ylim(0, 100)

# Acceleration Bars
y_2 = u_21_striker['Acceleration']

n = 2
t = 3
d = 28
w = 0.8
x_values2 = [t*element + w*n for element in range(d)]

plt.bar(x_values2, y_2, label='Acceleration')

# Agility Bars
y_3 = u_21_striker['Agility']

n = 3
t = 3
d = 28
w = 0.8
x_values3 = [t*element + w*n for element in range(d)]

plt.bar(x_values3, y_3, label='Agility')
plt.legend(loc='lower right')
#plt.show()

"""From the visualization above we can tell that the player with the ID '234045' is the most well rounded player when
it comes to speed, acceleration and agility. He has one of the fastest sprint speeds, one of the fastest
accelerations, and the best agility. When it comes to speed he is our top pick."""


"""Next I want to create a scatter plot to show the dribbling and ball control a player has. I want my pick
to be someone who is near the top right cluster here - someone with good ball control and dribbling."""

plt.figure(figsize=(10, 5))
ax = plt.subplot(1, 1, 1)
x = u_21_striker['BallControl']
y = u_21_striker['Dribbling']
n = u_21_striker['ID']
plt.scatter(x, y)
plt.xlabel('Ball Control')
plt.ylabel('Dribbling')

for i, txt in enumerate(n):
    plt.annotate(txt, (x[i], y[i]))

#plt.show()

"""The visualization from the code above shows us that ID '234045' and '225713' are the best at both
dribbling and ball control. They are both between 75-85 in both dribbling and ball control. '234045' is 
slightly better at dribbling than he is at ball control, while '225713' is slightly better at ball control
than he is at dribbling. The majority of players are in the middle of the pack - within the 60-75 range
for both dribbling and ball control. We see here that there is a direct linear relationship between
both dribbling and ball control"""


"""The last thing we need to look at is Finishing, HeadingAccuracy, 
ShotPower, Volleys, Curve, and LongShots. """

x = range(len(u_21_striker[u_21_striker['Finishing']>70]))
above_70_finishing = u_21_striker[u_21_striker['Finishing'] > 70]
y = above_70_finishing['Finishing']
figure = plt.figure(figsize=(10, 5))
ax = plt.subplot(1, 1, 1)

plt.bar(x, y)
plt.xlabel('Player ID')
plt.ylabel('Finishing Rating')
ax.set_xticks(x)
ax.set_xticklabels(above_70_finishing['ID'])
#plt.show()

#u_21_striker = u_21_striker[u_21_striker['ID'] == 234045]
#print(u_21_striker)

# Creating a function to calculate multiple x values for side by side bar plots

def multiple_bar_charts(n, t, d, w):
    n = n
    t = t
    d = d
    w = w
    x_values = [t*element + w*n for element in range(d)]
    return x_values

# Plotting a multiple bar chart for HeadingAccuracy, ShotPower, Volleys, Curve, and LongShots.

ax = plt.subplot(1, 1, 1)
x_values1 = multiple_bar_charts(1, 5, 28, 0.8)
y1 = u_21_striker['ShotPower']

plt.bar(x_values1, y1, label='Shot Power')


x_values2 = multiple_bar_charts(2, 5, 28, 0.8)
y2 = u_21_striker['Curve']

plt.bar(x_values2, y2, label='Curve')


x_values3 = multiple_bar_charts(3, 5, 28, 0.8)
y3 = u_21_striker['LongShots']

plt.bar(x_values3, y3, label='Long Shots')


x_values4 = multiple_bar_charts(4, 5, 28, 0.8)
y4 = u_21_striker['HeadingAccuracy']

plt.bar(x_values4, y4, label='Heading Accuracy')


x_values5 = multiple_bar_charts(5, 5, 28, 0.8)
y5 = u_21_striker['Volleys']

plt.bar(x_values5, y5, label='Volleys')
ax.set_xticks(x_values3)
ax.set_xticklabels(u_21_striker['ID'], rotation=90)
plt.legend()
#plt.show()

"""From looking at the data, the tables and the charts we see that the player with the ID '234045' also has one
 of the stornger goal scoring attributes. He has the best long shot, one of the best curves available, and one
 of the best volleys. The one thing he needs to work on the most is his heading accuracy. 
 
 Based on all the visualizations and data on hand player ID '234045' is our best option for starting Striker."""

u_21_striker = u_21_striker[u_21_striker['ID'] == 234045]


plt.close('all')
"""Next let's find our left and right wingers to compliment our striker. We need speedy players who have good
long shots, and are good at passing and crossing."""

u_21_wings = u_21[(u_21['Position'] == 'LW') | (u_21['Position'] == 'RW') & (u_21['SprintSpeed'] > 85)]
u_21_wings = u_21_wings.drop(u_21_wings.columns[28:45], axis=1).reset_index(drop=True)

"""The dataframe u_21_wings gives contains data on 153 players.
Let's create a new dataframe of the fastest of the fast when it comes to wingers."""

u_21_wings_fastest = u_21_wings[(u_21_wings['SprintSpeed'] >= 90) & (u_21_wings['Acceleration'] >= 90
                                                                    ) & (u_21_wings['Agility'] >= 90)]

"""This creates a DataFrame with the top 5 fastest players based on SprintSpeed, Acceleration and Agility.
Let's create a side by side bar chart to compare them."""
x_values1 = multiple_bar_charts(1, 3, 5, 0.8)
y = u_21_wings_fastest['SprintSpeed']

x_values2 = multiple_bar_charts(2, 3, 5, 0.8)
y2 = u_21_wings_fastest['Acceleration']

x_values3 = multiple_bar_charts(3, 3, 5, 0.8)
y3 = u_21_wings_fastest['Agility']

plt.figure(figsize=(10, 5))
ax_2 = plt.subplot(1, 1, 1)
plt.bar(x_values1, y, label='Sprint Speed')
plt.bar(x_values2, y2, label='Acceleration')
plt.bar(x_values3, y3, label='Agility')
plt.legend()
ax_2.set_xticks(x_values2)
ax_2.set_xticklabels(u_21_wings['ID'], rotation=90)
plt.xlabel('Player ID')
plt.ylabel('Rating')
plt.title("Player's Fastness")
#plt.show()

plt.close('all')

"""The top 5 fastest players are all around the same speed, and all are well rounded.
The ID's we have for top 5 are 231443, 238794, 235527, 241695, and 242841.
The top two overall rated players are 231443, 238794"""


"""Next let's look at the top passers. Let's create a new dataframe with players who have
ShortPassing, LongPassing and Crossing above 69."""
u_21_wings_best_passers = u_21_wings[(u_21_wings['Crossing'] > 69) & (u_21_wings['ShortPassing'] > 69
                                                                      ) & (u_21_wings['LongPassing'] > 69)]

x_values1 = multiple_bar_charts(1, 3, 2, 0.8)
y = u_21_wings_best_passers['Crossing']

x_values2 = multiple_bar_charts(2, 3, 2, 0.8)
y2 = u_21_wings_best_passers['ShortPassing']

x_values3 = multiple_bar_charts(3, 3, 2, 0.8)
y3 = u_21_wings_best_passers['LongPassing']

#ax_3 = plt.subplot()
#plt.bar(x_values1, y, label='Crossing')
#plt.bar(x_values2, y2, label='Short Passing')
#plt.bar(x_values3, y3, label='Long Passing')
#plt.xlabel('Player ID')
#plt.ylabel('Rating')
#ax_3.set_xticks(x_values2)
#ax_3.set_xticklabels(u_21['ID'], rotation=90)
#plt.title('Passing Rating')
#plt.legend()
#plt.show()

plt.close()

"""From the table and visualization we gather that there are only two players out of 153 who have ratings
above 69 in crossing, short passing, and long passing. Those are player ID 231443, and 220746.
220746 is a slightly better overall passer, but 231443 is better at crosses, and only slightly behind in 
both short and long passes.
Since we only got two players here, let's open it up more and bring back the dataframe qualifications to 60."""

u_21_wings_best_passers = u_21_wings[(u_21_wings['Crossing'] > 60) & (
        u_21_wings['ShortPassing'] > 60) & (u_21_wings['LongPassing'] > 60)].reset_index(drop=True)

"""Now we have 19 samples in this dataframe. Let's map these 19 out and try and make some more conclusions."""
x_values1 = multiple_bar_charts(1, 3, 20, 0.8)
y = u_21_wings_best_passers['Crossing']

x_values2 = multiple_bar_charts(2, 3, 20, 0.8)
y2 = u_21_wings_best_passers['ShortPassing']

x_values3 = multiple_bar_charts(3, 3, 20, 0.8)
y3 = u_21_wings_best_passers['LongPassing']

ax_3 = plt.subplot()
plt.bar(x_values1, y, label='Crossing')
plt.bar(x_values2, y2, label='Short Passing')
plt.bar(x_values3, y3, label='Long Passing')
plt.xlabel('Player ID')
plt.ylabel('Rating')
ax_3.set_xticks(x_values2)
ax_3.set_xticklabels(u_21['ID'], rotation=90)
plt.title('Passing Rating')
plt.legend()
#plt.show()

plt.close('all')

"""From this we gather the best passers are 231443, 231677, 220746, 228819, 241852, 240802.
Next let's look at long shots."""

u_21_wings_shots = u_21_wings[(u_21_wings['ShotPower'] > 70) & (u_21_wings['LongShots'] > 70)]\
    .reset_index(drop=True)

#print(u_21_wings_shots)

x1 = u_21_wings_shots['ShotPower']
y1 = u_21_wings_shots['LongShots']
n1 = u_21_wings_shots['ID']
plt.scatter(x1, y1)
plt.xlabel('Shot Power')
plt.ylabel('Long Shot Accuracy')
plt.title('Player Shooting ratings')
for i, txt in enumerate(n1):
    plt.annotate(txt, (x1[i], y1[i]))


#plt.show()
plt.close()

"""From this we see that Player ID 231677 by far has the best long shot rating. Player 215798 has the second
 best long shot, but he lacks in power and accuracy compared to player 231677."""


"""From all the info above we have one player who is a leader in speed and passing, and another player
who has good speed, good passing, and is the best at long distance shots. These will be our two wingers.
I will store them in variables below."""

u_21_lw = u_21_wings[u_21_wings['ID'] == 231677]

u_21_rw = u_21_wings[u_21_wings['ID'] == 231443]

u_21_attacking_mids = u_21[(u_21['Position'] == 'LAM') | (u_21['Position'] == 'RAM') | (u_21['Position'] == 'RM')
                           | (u_21['Position'] == 'LM') | (u_21['Position'] == 'CAM')
                           | (u_21['Position'] == 'LCM') | (u_21['Position'] == 'RCM')
                           | (u_21['Position'] == 'CM')]


u_21_attacking_mids = u_21_attacking_mids.drop(u_21_attacking_mids.columns[19:27], axis=1)

u_21_attacking_mids = u_21_attacking_mids.drop(u_21_attacking_mids.columns[27:37], axis=1)\
    .reset_index(drop=True)

u_21_attacking_mids = u_21_attacking_mids[u_21_attacking_mids['SprintSpeed'] > 85].reset_index(drop=True)

#print(u_21_attacking_mids)

"""We setup our attacking mids dataframe with the code above. 
For an attacking mid the most important attributes to measure is Crossing, ShortPassing, Dribbling, LongPassing,
Ball Control, Stamina, and Vision.
Let's visualize some of our players below"""

"""Let's start by measuring the passing stats of players. I want players who are above a 60 in crossing, 
above 70 in short passing, and above 60 long passing"""

u_21_attacking_mids_passers = u_21_attacking_mids[(u_21_attacking_mids['Crossing'] > 60) &\
                                                  (u_21_attacking_mids['ShortPassing'] > 70) &\
                                                  (u_21_attacking_mids['LongPassing'] > 60)].reset_index(drop=True)

#print(u_21_attacking_mids_passers)

ax = plt.subplot(1, 1, 1)
x_values1 = multiple_bar_charts(1, 3, 9, 0.8)
y = u_21_attacking_mids_passers['Crossing']

x_values2 = multiple_bar_charts(2, 3, 9, 0.8)
y_2 = u_21_attacking_mids_passers['ShortPassing']

x_values3 = multiple_bar_charts(3, 3, 9, 0.8)
y_3 = u_21_attacking_mids_passers['LongPassing']

plt.bar(x_values1, y, label='Crossing')
plt.bar(x_values2, y_2, label='ShortPassing')
plt.bar(x_values3, y_3, label='LongPassing')
ax.set_xticks(x_values2)
ax.set_xticklabels(u_21_attacking_mids_passers['ID'])
plt.xlabel('ID')
plt.ylabel('Passer Ratings')
plt.legend()
#plt.show()


plt.close('all')

"""From the data we determine that 231747 is the best midfield passer. He is tied for best short passing,
he is the best at crossing, and he is second best long passer.

Other things determined - 229261 is the best long passer, and tied for best short passer, but he is only 
a 64 at crossing.

233550 is very well rounded amongst all three - crossing, short passing and long passing."""


"""Next let's  plot Ball Control and Dribbling"""
x = u_21_attacking_mids['BallControl']
y = u_21_attacking_mids['Dribbling']
n = u_21_attacking_mids['ID']

x_matrix = x.values.reshape(-1, 1)
kmeans = KMeans(3)
kmeans.fit(x_matrix)
# I want to create a cluster to differentiate different levels of ball control
clusters = kmeans.fit_predict(x_matrix)
u_21_attacking_mids['Clusters'] = clusters

ax = plt.subplot(1, 1, 1)
plt.scatter(x, y, marker="o", c=clusters, cmap='rainbow')
plt.xlabel('Ball Control')
plt.ylabel('Dribbling')
for i, txt in enumerate(n):
    plt.annotate(txt, (x[i], y[i]))

#plt.show()

plt.close()

"""From this visualization we cluster the following ID's in the best ball handler/ dribbling cluster:
231747, 227796, 233049, 229906, 224411, 235790, 233631, 229453, 244193, 239015, 233419, 235353, 235883, 229261.

The best ball handler/ dribbler is 231747.
The 4 best after him are 227796, 233049, 229906, 224411"""

u_21_attacking_mids_dribblers = u_21_attacking_mids[(u_21_attacking_mids['BallControl'] > 80) & \
                                                    (u_21_attacking_mids['Dribbling'] > 80)]


"""Let's look at a distribution of our attacking mid's visions"""
plt.hist(u_21_attacking_mids['Vision'])
#plt.show()
plt.close()

"""Based off of that let's create a dataframe with only the attacking mids with the highest visions"""
u_21_attacking_mids_vision = u_21_attacking_mids[u_21_attacking_mids['Vision'] > 75]
#print(u_21_attacking_mids_vision)


"""Lastly let's look at player Stamina's"""
plt.hist(u_21_attacking_mids['Stamina'])
#plt.show()

plt.close('all')

u_21_attacking_mids_stamina = u_21_attacking_mids[u_21_attacking_mids['Stamina'] >= 80]
#print(u_21_attacking_mids_stamina)

u_21_rm = u_21_attacking_mids[u_21_attacking_mids['ID'] == 231747]
u_21_lm = u_21_attacking_mids[u_21_attacking_mids['ID'] == 229906]

"""We chose player ID 231747 as our Right Midfielder, because statistically in all the important stat
categories he is by far the best available midfielder.

We chose player ID 229906 based on a few other factors. This player ranked high in passing and ball handling/
dribbling, while also having average positioning and average stamina. It was a close decision between this
and player ID '', but there were a few other factors like age, foot, and salary that made the difference.
Player ID 229906 was one year younger, was left footed, and was cheaper - not to mention he was a slightly
faster player too."""


"""Now let's establish our Defensive midfielder.
We want someone who can pass the ball well, but who also can tackle well and has good 
positioning and vision. A majority of the available defensive midfielders are slower, so we will adjust
our SprintSpeed cutoff to players whose SprintSpeed is greater than 70."""

u_21_defensive_mid = u_21[(u_21['Position'] == 'LDM') | (u_21['Position'] == 'CDM') |\
                          (u_21['Position'] == 'RDM')].reset_index(drop=True)

ax = plt.subplot(1,2,1)
plt.hist(u_21_defensive_mid['StandingTackle'])
plt.xlabel('Standing Tackle Rating')
plt.title('Standing Tackle Distribution')
ax_2 = plt.subplot(1,2,2)
plt.hist(u_21_defensive_mid['SlidingTackle'])
plt.xlabel('Sliding Tackle Rating')
plt.title('Sliding Tackle Distribution')
#plt.show()

plt.close()

u_21_defensive_mid = u_21_defensive_mid[u_21_defensive_mid['SprintSpeed'] > 70].reset_index(drop=True)

u_21_dm_best_st_tacklers = u_21_defensive_mid[u_21_defensive_mid['StandingTackle'] > 70]
u_21_dm_best_sl_tacklers = u_21_defensive_mid[u_21_defensive_mid['SlidingTackle'] > 70]


"""I created two new dataframes above which give us the top standing tacklers, and sliding tacklers.
From these dataframes we see only player 226790 and 235997 are in both dataframes. 
Player 228702 missed being in the sliding tackler dataframe by 1 overall point.
From the dataframes above we can say with confidence player 226790 is the best tackler.
Let's look at the passing abilities"""

u_21_defensive_mid_passers = u_21_defensive_mid[(u_21_defensive_mid['LongPassing'] > 70) |
                                                (u_21_defensive_mid['ShortPassing'] > 70)].reset_index(drop=True)




x = u_21_defensive_mid_passers['ShortPassing']
y = u_21_defensive_mid_passers['LongPassing']
n = u_21_defensive_mid_passers['ID']
plt.scatter(x, y)
plt.xlabel('Short Passing')
plt.ylabel('Long Passing')
for i, txt in enumerate(n):
    plt.annotate(txt, (x[i], y[i]))
#plt.show()
plt.close()

"""The graph shows us how much better of a passer 228702 is from the rest of the group. 
The majority of the spread of the data is withi 70 - 75 overall for both passing attributes."""

"""Lets take a look at the vision of our defensive midfielders. We'll start by plotting a 
histogram to take a look at the spread of the data."""
plt.hist(u_21_defensive_mid['Vision'])
plt.xlabel('Rating')
plt.ylabel('Frequency')
#plt.show()
plt.close()

mid_vision_mean = np.mean(u_21_defensive_mid['Vision'])
mid_vision_std = np.std(u_21_defensive_mid['Vision'])

"""It looks like we have a left skewed data set. The mean is around 59 overall, with a std of 10.
So 95% of the data is between 39 - 79 overall.
Let's take a look at only the players above 69% or within the upper 2nd standard deviation"""
u_21_defensive_mid_vision = u_21_defensive_mid[u_21_defensive_mid['Vision'] > 69]

"""Player ID 228702 has the best vision out of any players. He is the only outlier from two
standard deviations from the mean, and is above the 95 percentile. He is also 11 overall points
higher than the second best player in terms of vision."""

"""Let's take a look at positioning. We want our Defensive midfielder to have good positioning
both to help us score goals, but also to prevent the other team from scoring."""
plt.hist(u_21_defensive_mid['Positioning'])
plt.xlabel('Rating')
plt.ylabel('Frequency')
#plt.show()
plt.close()

"""We see a normal distribution here, with the highest rated positioning player being between 70-80.
Let's narrow down our search to players who are above 65 in terms of positioning"""
u_21_defensive_mid_positioning = u_21_defensive_mid[u_21_defensive_mid['Positioning'] > 65]

"""Player ID 244264 is the best ranked positioning player, but out of the top ranked players in positioning, 
he has the lowest overall rating.
Player ID 226790 is the third best ranked positioning player, but is the best overall ranked player."""

"""Let's also look at stamina"""
plt.hist(u_21_defensive_mid['Stamina'])
plt.xlabel('Rating')
plt.ylabel('Frequency')
#plt.show()
plt.close()

u_21_defensive_mid_stamina = u_21_defensive_mid[u_21_defensive_mid['Stamina'] > 75]
#print(u_21_defensive_mid_stamina)

"""We conclude player ID 226790 has the best Stamina"""

"""Let's scale all of these weights to see which are the heaviest / most important in order to help us make our
decision"""


u_21_defensive_mid['CDM'] = u_21_defensive_mid['CDM'].apply(lambda x: x.split('+')[0])
u_21_defensive_mid['CDM'] = u_21_defensive_mid['CDM'].astype(int)

x = u_21_defensive_mid[['LongPassing', 'StandingTackle', 'SlidingTackle',
                        'Positioning', 'Vision', 'Stamina', 'ShortPassing']]
y = u_21_defensive_mid['CDM']



scaler = StandardScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)

reg = LinearRegression()
reg.fit(x_scaled, y)

confusion_matrix = pd.DataFrame(data=x.columns)
confusion_matrix['Weights'] = reg.coef_

best_mid_fielder = 1.982479*u_21_defensive_mid['ShortPassing'] + 0.660455*u_21_defensive_mid['LongPassing']\
    + 2.845148*u_21_defensive_mid['StandingTackle'] + 0.519853*u_21_defensive_mid['SlidingTackle']\
    + 0.141193*u_21_defensive_mid['Positioning'] + 0.504530*u_21_defensive_mid['Vision']\
    + 1.404681*u_21_defensive_mid['Stamina']

u_21_defensive_mid['New Overall'] = best_mid_fielder

#print(u_21_defensive_mid.sort_values('New Overall', ascending=False))

"""By scaling the features, and building a regression model using the features to
obtain position overall targets, we determine which features are the most important. 
Then we use the coefficients to calculate the total value of each feature and store that in a new
column in the main u_21_defensive_mid DataFrame titled 'New Overall'. From this we see that
player ID 226790 has the highest total number, and therefore is the best choice for
as our central defensive midfielder."""

u_21_defensive_mid = u_21_defensive_mid[u_21_defensive_mid['ID'] == 226790]

"""Let's start building a dataframe for LB and RB. What I want to do is only include all the relevant
columns pertaining to LB and RB overall rating. Let's find all RB's who """

u_21_rb_lb = u_21[(u_21['Position'] == 'RB') | (u_21['Position'] == 'RWB')\
               | (u_21['Position'] == 'LB') | (u_21['Position'] == 'LWB') & (u_21['SprintSpeed'] > 70)]
u_21_rb_lb = u_21_rb_lb[['ID', 'Name', 'Overall', 'Potential', 'Value', 'Preferred Foot', 'Weak Foot', 'Skill Moves'\
                , 'Position', 'RB', 'LB', 'Crossing', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'LongPassing'\
                , 'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance'\
                , 'Jumping', 'Stamina', 'Strength', 'Aggression', 'Interceptions'\
                , 'Positioning', 'Vision', 'Marking', 'StandingTackle', 'SlidingTackle']]

x = u_21_rb_lb.drop(['ID', 'RB', 'LB', 'Name', 'Position', 'Value', 'Preferred Foot'], axis=1)
u_21_rb_lb['RB'] = u_21_rb_lb['RB'].apply(lambda x: x.split('+')[0])
u_21_rb_lb['RB'] = u_21_rb_lb['RB'].astype(int)
u_21_rb_lb['LB'] = u_21_rb_lb['LB'].apply(lambda x: x.split('+')[0])
u_21_rb_lb['LB'] = u_21_rb_lb['LB'].astype(int)

y_rb = u_21_rb_lb['RB']
y_lb = u_21_rb_lb['LB']

scaler = StandardScaler()
scaler.fit(x, y_rb)
x_scaled = scaler.transform(x)

reg = LinearRegression()
reg.fit(x_scaled, y_rb)
reg.score(x_scaled, y_rb)

"""From checking the R-squared, it seems our model explains 99% of the variability!
Let's check the F-Stat to see if our model is a good fit"""

f_test, _ = f_regression(x_scaled, y_rb)
f_test = np.max(f_test)

"""The F-Test max is 29,530 - it seems from this that our model is a very good fit."""

df = pd.DataFrame(data=x.columns)
df['Weights'] = reg.coef_.round(3)
df['Bias'] = reg.intercept_.round(3)
#print(df)

"""Now that we have our coefficients and bias, let's calculate each players rating based on our model"""

results = 0.003*u_21_rb_lb['Weak Foot'] - 0.035*u_21_rb_lb['Skill Moves'] + 0.789 * u_21_rb_lb['Crossing']\
    + 0.809*u_21_rb_lb['ShortPassing'] + 0.03*u_21_rb_lb['Volleys'] + 0.011*u_21_rb_lb['Dribbling']\
    - 0.041*u_21_rb_lb['LongPassing'] + 0.654*u_21_rb_lb['BallControl'] + 0.28*u_21_rb_lb['Acceleration']\
    + 0.599*u_21_rb_lb['SprintSpeed'] - 0.046*u_21_rb_lb['Agility'] + 0.549 * u_21_rb_lb['Reactions']\
    - 0.022*u_21_rb_lb['Balance'] - 0.001*u_21_rb_lb['Jumping'] + 0.669*u_21_rb_lb['Stamina']\
    + 0.052*u_21_rb_lb['Strength'] + 0.009*u_21_rb_lb['Aggression'] + 0.826*u_21_rb_lb['Interceptions']\
    + 0.020*u_21_rb_lb['Positioning'] - 0.015*u_21_rb_lb['Vision'] + 0.576*u_21_rb_lb['Marking']\
    + 0.791*u_21_rb_lb['StandingTackle'] + 0.998*u_21_rb_lb['SlidingTackle'] + 59.483

u_21_rb_lb['Total Points'] = results
#print(u_21_rb_lb)

"""From our conclusion above we find that the highest rated LB is player ID '235212' and the 
highest rated RB is player ID '221342', so these two players are our choices."""

u_21_rb = u_21[u_21['ID'] == 221342]
u_21_lb = u_21[u_21['ID'] == 221342]

"""Next let's choose our two CB's."""
u_21_cbs = u_21[(u_21['Position'] == 'CB') & (u_21['SprintSpeed'] > 70)].reset_index(drop=True)

u_21_cbs = u_21_cbs[['ID', 'Name', 'Overall', 'Potential', 'Preferred Foot', 'CB', 'HeadingAccuracy', 'ShortPassing'\
                     , 'LongPassing', 'BallControl', 'SprintSpeed', 'Acceleration', 'Balance'\
                     , 'Jumping', 'Stamina', 'Strength', 'Aggression', 'Interceptions', 'Positioning'\
                     , 'Marking', 'StandingTackle', 'SlidingTackle']]

u_21_cbs['CB'] = u_21_cbs['CB'].apply(lambda x: x.split('+')[0])
u_21_cbs['CB'] = u_21_cbs['CB'].astype(int)

x = u_21_cbs.drop(['ID', 'Name', 'Overall', 'Potential', 'Preferred Foot', 'CB'], axis=1)
y = u_21_cbs['CB']

scaler = StandardScaler()
scaler.fit(x, y)
x_scaled = scaler.transform(x)

reg = LinearRegression()
reg.fit(x, y)
#print(reg.score(x, y))
f_stat, _ = f_regression(x, y)
f_stat = np.max(f_stat)
#print(f_stat)

"""Our model seems to explain 99.75% of the variability of the CB overall rating!
The F-Stat is a high number of 486.90, which tells us that our model looks significant."""

df = pd.DataFrame(x.columns)
df['Weights'] = reg.coef_.round(3)
df['Bias'] = reg.intercept_.round(3)
#print(df)

results = 0.127*u_21_cbs['HeadingAccuracy'] + 0.052*u_21_cbs['ShortPassing'] - 0.009*u_21_cbs['LongPassing']\
    + 0.055*u_21_cbs['BallControl'] + 0.044*u_21_cbs['SprintSpeed'] + 0.006*u_21_cbs['Acceleration']\
    + 0.006*u_21_cbs['Balance'] + 0.024*u_21_cbs['Jumping'] + 0.001*u_21_cbs['Stamina']\
    + 0.098*u_21_cbs['Strength'] + 0.083*u_21_cbs['Aggression'] + 0.141*u_21_cbs['Interceptions']\
    - 0.004*u_21_cbs['Positioning'] + 0.145*u_21_cbs['Marking'] + 0.205*u_21_cbs['StandingTackle']\
    + 0.058*u_21_cbs['SlidingTackle'] - 3.071

u_21_cbs['Total Rating'] = results

#print(u_21_cbs.sort_values('Total Rating', ascending=False))


"""From the analysis of the model above we conclude that our two best CB's are player ID 225100,
and player ID 225161."""

u_21_cbs = u_21_cbs[(u_21_cbs['ID'] == 225100) | (u_21_cbs['ID'] == 225161)]

u_21_gk = u_21[u_21['Position'] == 'GK'].reset_index()
u_21_gk = u_21_gk[['ID', 'Name', 'Overall', 'Potential', 'Position', 'GKDiving',
                    'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']]

x = u_21_gk[['GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']]
y = u_21_gk['Overall']

scaler = StandardScaler()
scaler.fit(x, y)
x_scaled = scaler.transform(x)
reg = LinearRegression()
reg.fit(x_scaled, y)

df = pd.DataFrame(data=x.columns)
df['Weights'] = reg.coef_.round(3)
df['Bias']= reg.intercept_.round(3)


#print(df)

results = 1.448*u_21_gk['GKDiving'] + 1.214*u_21_gk['GKHandling'] + 0.289*u_21_gk['GKKicking']\
    + 1.594*u_21_gk['GKPositioning'] + 1.698*u_21_gk['GKReflexes'] + 57.808

u_21_gk['Total Rating'] = results

#print(u_21_gk)


u_21_gk = u_21_gk[u_21_gk['ID'] == 230621]
#print(u_21_gk)


final_team = pd.DataFrame()
final_team['Position'] = ['ST', 'RW', 'LW', 'RAM', 'LAM', 'CDM', 'RB', 'CB', 'CB', 'LB', 'GK']
final_team['Player'] = [u_21_striker['Name'].values, u_21_rw['Name'].values, u_21_lw['Name'].values,
                        u_21_rm['Name'].values, u_21_lm['Name'].values, u_21_defensive_mid['Name'].values,
                        u_21_rb['Name'].values, u_21_cbs['Name'][0], u_21_cbs['Name'][1],
                        u_21_lb['Name'].values, u_21_gk['Name'].values]

final_team['Overall'] = [u_21_striker['Overall'].values, u_21_rw['Overall'].values, u_21_lw['Overall'].values,
                        u_21_rm['Overall'].values, u_21_lm['Overall'].values, u_21_defensive_mid['Overall'].values,
                        u_21_rb['Overall'].values, u_21_cbs['Overall'][0], u_21_cbs['Overall'][1],
                        u_21_lb['Overall'].values, u_21_gk['Overall'].values]

final_team['Potential'] = [u_21_striker['Potential'].values, u_21_rw['Potential'].values, u_21_lw['Potential'].values,
                        u_21_rm['Potential'].values, u_21_lm['Potential'].values, u_21_defensive_mid['Potential'].values,
                        u_21_rb['Potential'].values, u_21_cbs['Potential'][0], u_21_cbs['Potential'][1],
                        u_21_lb['Potential'].values, u_21_gk['Potential'].values]

final_team['SprintSpeed'] = [u_21_striker['SprintSpeed'].values, u_21_rw['SprintSpeed'].values, u_21_lw['SprintSpeed'].values,
                        u_21_rm['SprintSpeed'].values, u_21_lm['SprintSpeed'].values, u_21_defensive_mid['SprintSpeed'].values,
                        u_21_rb['SprintSpeed'].values, u_21_cbs['SprintSpeed'][0], u_21_cbs['SprintSpeed'][1],
                        u_21_lb['SprintSpeed'].values, 'NaN']

final_team_speed_mean = np.mean(final_team['SprintSpeed'][0:10])
final_team_potential_mean = np.mean(final_team['Potential'])
final_team_overall_mean = np.mean(final_team['Overall'])

"""The DataFrame final_team shows our final team.
We see the average Sprint Speed for our team is 86
The average potential is 89
The average overall is 80

Not bad for a team of kids all under 21 years old!"""