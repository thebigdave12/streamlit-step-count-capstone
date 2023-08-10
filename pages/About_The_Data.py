import streamlit as st;
import pandas as pd;
import seaborn as sns;
import matplotlib.pyplot as plt 

data = pd.read_csv('fitbit_mean_miles_df.csv')
data_flushed = pd.read_csv('fitbit_flushed_miles_df.csv')
data_og = pd.read_csv('fitbit_data.csv')

st.title('About the Data:')
st.markdown("***")
st.markdown('The data set used for this project comes from a fitness study that looked at 32 participants who actively wore Fitbit trackers over the course of a month. The participants came from a sample of mixed ages, fitness levels, and gender. Thus, it is representative of an active community of people who care about measuring and taking care of their health. For more details on the data set feel free to visit:')
st.markdown("https://www.kaggle.com/datasets/gloriarc/fitbit-fitness-tracker-data-capstone-project")
st.markdown("***")

st.markdown("<div style='text-align: center;'> <h2> By the Numbers </h2> </div>", unsafe_allow_html=True)

x = data.TotalSteps
y = data.Calories 

plt.figure(figsize=(10, 6))
plt.scatter(x, y)
plt.xlabel('Total Steps', fontsize=20)
plt.ylabel('Calories', fontsize=20)
st.pyplot(plt)

col1, col2, col3 = st.columns(3)

col1.metric("Avg. Daily Step Count", "8,253")
col2.metric("Number of Participants", "32")
col3.metric("Total Daily Log (sample size)", "940")

st.markdown("***")

st.markdown("<div style='text-align: center;'> <h2> What was tracked in the study? </h2> </div>", unsafe_allow_html=True)
st.markdown("""
- Date of Log
- Daily Step Count
- Daily Distance by Mile
- Daily Total of Active minutes (very active, fairly active, lightly active)
- Daily Sedentary Minutes (time inactive)
- Daily Calories Burned
""")
            
st.markdown("***")           
st.markdown("<div style='text-align: center;'> <h2> Transformations Made to Data Set: </h2> </div>", unsafe_allow_html=True)

# Original Data Distribution Chart:
plt.figure(figsize=(12, 12))
ax_og = sns.displot(data_og['TotalSteps'], kde=True)
plt.ylabel('Number of Logs')
plt.xlabel('Total Steps')
plt.title('Distribution of Step Count by day')
st.pyplot(plt)

st.markdown('Upon exploring the original Fitbit data set I quickly found that there were a high number of logs with a very low step count. As you can see from the distribution graph above, this caused an irregular distribution, starting high and ending low (not a normal bell curve). After deeper exploration of the experiment, I determined that many of the low step count logs were due to a need to charge the Fitbit wearable. Oftentimes, participants would forget or choose not to wear their Fitbits for the rest of the day if it was a “charge day”. Thus, the 90 logs with an irregularly low step count were throwing off the rest of the data…')
st.markdown('In order to fix this error in distribution, I performed two transformations.  The first being I took the average of all the data with accurate information (above 500 step count) and inserted those means into all the columns with irregular date (below 500 step count). As you can see below, this causes a massive number of logs in the middle of the distribution, but it does transform the distribution into a normal bell curve. The second transformation was a simple dropping of all rows with irregular step count values. This decreases the overall sample size but created the most normal distribution. In the end, both transformations led to nearly identical linear models. I ended up picking the mean data frame because it had a higher f-statistic showing that it is more likely a better fit for predictions.')
st.markdown('')
col1, col2 = st.columns(2)

# Mean Data Distribution Chart:
plt.figure(figsize=(6, 8))
ax_mean = sns.displot(data['TotalSteps'], kde=True)
plt.ylabel('Number of Logs')
plt.xlabel('Total Steps')
plt.title('Step Count by day transformed w/ mean data)')
col1.pyplot(plt)

# Flushed Data Distribution Chart:
plt.figure(figsize=(12, 12))
ax_flushed = sns.displot(data_flushed['TotalSteps'], kde=True)
plt.ylabel('Number of Logs')
plt.xlabel('Total Steps')
plt.title('Step Count by day transformed w/ dropped data')
col2.pyplot(plt)
st.markdown("***")

st.markdown("<div style='text-align: center;'> <h2> Other things to note about Fitbit: </h2> </div>", unsafe_allow_html=True)
st.markdown("<div style='display: flex; justify-content: center;'>"
    "<img src='https://miro.medium.com/v2/resize:fit:1400/1*Rm5stTE51nPk7WxXhC6JDA.jpeg' style='height: 650px; display: block;'>"
    "</div>", unsafe_allow_html=True)

st.markdown("<div style='text-align: center;'> <h2> How does Fitbit calculate calories burned? </h2> </div>", unsafe_allow_html=True)

st.markdown('So you may be wondering, how does Fitbit calculate calories burned? Fitbit devices combine your basal metabolic rate (BMR)—the rate at which you burn calories at rest to maintain vital body functions (including breathing, blood circulation, and heartbeat)—and your activity data to estimate your calories burned. Heart rate is key to estimating calories burned during exercise. The number you see on your Fitbit device is your total calories burned for the day.')
st.markdown('Your BMR is based on the physical data you entered your Fitbit account (height, weight, sex, and age) and accounts for at least half the calories you burn in a day. Because your body burns calories even if you’re asleep or not moving, you see calories burned on your device when you wake up and will notice this number increase throughout the day.')
st.markdown('With that being said, it will be interesting to see the correlation between calories burned and metrics reported by participants. All in all, we should be able to make a somewhat accurate prediction of calories burned based off of step count.')

st.markdown("<div style='text-align: center;'> <h2> So what other health metrics does Fitbit Track? </h2> </div>", unsafe_allow_html=True)
st.markdown("""
- Sleep 
- Temperature
- Heart Rate
- Floors climbed
- Specific type of exercise (Bike, treadmill, hike, etc.)
""")
            
st.markdown("<div style='text-align: center;'> <h2> Predictive Data: </h2> </div>", unsafe_allow_html=True)
st.markdown("In order to test the models, I built out a data frame with 7 months of my own fitness data tracked by fitbit. The data frame includes the following Metrics:")
st.markdown("""
- Date of Log
- Step Count
- Total Distance
- Active Minutes
- Calories Burned
""")

st.markdown("***") 
st.markdown("<div style='text-align: center;'> <h2> Take a look at the data dictionary: </h2> </div>", unsafe_allow_html=True)  

st.markdown("https://docs.google.com/document/d/1i1pThbEJZuE9GVgRLM0M26viCWBhJCBkHgmOoi6ZwkQ/edit?usp=sharing")