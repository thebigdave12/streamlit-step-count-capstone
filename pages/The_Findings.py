import streamlit as st;
import pandas as pd;
import seaborn as sns;
import matplotlib.pyplot as plt 
import statsmodels.api as sm

data = pd.read_csv('fitbit_mean_miles_df.csv')
dave_data = pd.read_csv('Dave_fitbit_stats.csv')
stepcount_prediction_df = pd.read_csv('stepcount_prediction_df.csv')
totaldistance_prediction_df = pd.read_csv('totaldistance_prediction_df.csv')
activeminutes_prediction_df = pd.read_csv('activeminutes_prediction_df.csv')
multivariate_prediction_df = pd.read_csv('multivariate_prediction_df.csv')

totals_by_id_df = data.groupby('Id')[["TotalSteps", "TotalDistance", "Calories", 'VeryActiveMinutes', 'FairlyActiveMinutes', 'LightlyActiveMinutes', "SedentaryMinutes"]].sum()
st.title('Fitbit Findings:')
with st.expander("Objective"):
    st.markdown("Hypothesis")
    st.markdown("Step Count is an accurate indicator of calories burned and in turn weight loss.")
    st.markdown("***")
    st.markdown("The Plan:")
    st.markdown("With our goal being to determine whether step count is a good indicator of weight loss, we will first start with a simple EDA looking at distribution, outliers, and other tracked metrics that could be better indicators of weight loss/health (distance, very active minutes, etc). ")
    st.markdown("***")
    st.markdown("Quick Look at the Data Set")
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg. Daily Step Count", "8,253")
    col2.metric("Number of Participants", "32")
    col3.metric("Total Daily Log (sample size)", "940")

    st.markdown("***")
with st.expander("Distributions"):
    st.markdown("Distributions:")
    st.markdown("I first started by looking at distributions of each data point to verify that we had normal distributions and no absurd outliers. I then merged each of the categories (step count, total distance, active minutes) on ID to see if we had any outlier participants. All in all, there were several participants that were clearly more active than the average Fitbit user, but nothing that was alarming or seemed out of the ordinary. With that information, I was able to move onto the next phase of the EDA correlations.")
    st.markdown("Note: In the distributions below, you'll see large spikes near the middle or top of the bell curve. This is because I replaced irregular data with the mean in order to maintain a large sample size (for more details see about the data).")
    col1, col2, col3 = st.columns(3)
    #Distribution of Step Count By Day (mean data)
    figure = plt.figure(figsize=(12,12))

    ax = sns.displot(data['TotalSteps'], kde=True)

    plt.ylabel('Number of Logs')
    plt.xlabel('Total Steps')
    plt.title('Distribution of Step Count by day (mean data)')
    col1.pyplot(plt)
    #Distribution of Total Distance (miles) By Day (mean data)
    figure = plt.figure(figsize=(12,12))

    ax = sns.displot(data['TotalDistance'], kde=True)

    plt.ylabel('Number of Logs')
    plt.xlabel('Total Distance')
    plt.title('Distribution of Total Distance by day (mean data)')
    col2.pyplot(plt)
    #Distribution of Very Active Minutes By Day (mean data)
    figure = plt.figure(figsize=(12,12))

    ax = sns.displot(data['VeryActiveMinutes'], kde=True)

    plt.ylabel('Number of Logs')
    plt.xlabel('Total Very Active Minutes')
    plt.title('Distribution of Very Active Minutes by day (mean data)')
    col3.pyplot(plt)
    #Distribution of Calories Burned By Day (mean data)
    figure = plt.figure(figsize=(12,12))

    ax = sns.displot(data['Calories'], kde=True)

    plt.ylabel('Number of Logs')
    plt.xlabel('Calories Burned')
    plt.title('Distribution of Calories Burned by day (mean data)')
    st.pyplot(plt)

    st.markdown("As we can see from the distributions, there tend to be a few outliers as the number increase creating longer tails. However, there is nothing irregular or misleading that would cause a need for another transformation or to toss out the data. One interesting thing to note is that the distribution of step count and total distance is nearly identical, meaning that both should have high correlations and be good indicators of step count. To get a better idea of the participants in the sample, we can look at distributions of the totals by participant of each health metric category.")



    col1, col2 = st.columns(2)

    #Distribution of total step count by participant 
    figure = plt.figure(figsize=(12,12))

    ax = sns.displot(totals_by_id_df['TotalSteps'], kde=True)

    plt.ylabel('Number of Participants')
    plt.xlabel('Total Steps over Test')
    plt.title('Distribution of total steps by Participant')
    col1.pyplot(plt)
    #Distribution of total distance by participant
    figure = plt.figure(figsize=(12,12))
    ax = sns.displot(totals_by_id_df['TotalDistance'], kde=True)
    plt.ylabel('Number of Participants')
    plt.xlabel('Total Distance over the course of Experiment')
    plt.title('Distribution of Total Distance by Participant')
    col2.pyplot(plt)

    st.markdown("As you can see from the chart below, there was one participant with an alarmingly low step count/total distance over the course of the study. Because of this, I decided to inspect further by creating a count plot by id which can be seen below. With this infograph we can clearly see that there was not faulty data but a lack of logs. Thus, no need to scrap, and it can be used for the regressions.")

    # Count plot for particpant # of daily logs
    participant_counts = data['Id'].value_counts()
    ordered_participant_ids = participant_counts.index
    plt.figure(figsize=(40, 10))  
    sns.countplot(data=data, x="Id", order=ordered_participant_ids)
    plt.ylabel('Number of daily logs')
    plt.xlabel('Participant ID')
    plt.title('Number of daily logs by participant')
    st.pyplot(plt)



    col1, col2, col3= st.columns(3)

    #Distribution of Lightly Minutes by participant
    figure = plt.figure(figsize=(12,12))

    ax = sns.displot(totals_by_id_df['LightlyActiveMinutes'], kde=True)

    plt.ylabel('Number of Participants')
    plt.xlabel('Total Lightly Active Minutes over Experiment')
    plt.title('Distribution of Lightly Active Minutes by Participant')
    col1.pyplot(plt)


    #Distribution of Very Active Minutes by participant
    figure = plt.figure(figsize=(12,12))

    ax = sns.displot(totals_by_id_df['FairlyActiveMinutes'], kde=True)

    plt.ylabel('Number of Participants')
    plt.xlabel('Total Fairly Active Minutes over Experiment')
    plt.title('Distribution of Fairly Active Minutes by Participant')
    col2.pyplot(plt)

    #Distribution of Very Active Minutes by Participant
    figure = plt.figure(figsize=(12,12))

    ax = sns.displot(totals_by_id_df['VeryActiveMinutes'], kde=True)

    plt.ylabel('Number of Participants')
    plt.xlabel('Total Very Active Minutes over Experiment')
    plt.title('Distribution of Very Active Minutes by Participant')
    col3.pyplot(plt)

    st.markdown("This final trio of distributions is to take a look at the activity levels of particpants. Based upon the lightly, fairly, and very active minute categories (see data dictionary for more details) we can see that the majority of the particpants exercised regularly. Some of the top outliers appear to be distance runners, which is why they are far ahead on very active minutes.")
    st.markdown("***")
with st.expander("Correlations"):
    st.markdown("Correlation:")
    st.markdown("After the distributions had been performed, I then was able to look at correlations between each category and calories burned. This would help us understand if there were any clear winners for predicting weight loss across all the categories. However, we must remember that correlation does not mean causation. Thus, we need to look at the models and perform testing in order to really say whether step count, total distance or active minutes are good indicators of calories burned and in turn cause weight loss.")
    #Heatmaps 
    heat_map_chart_mean_df = data[['TotalSteps', "TotalDistance", "VeryActiveMinutes", "Calories"]]
    corr_matrix = heat_map_chart_mean_df.corr()
    fig = plt.figure()
    plt.title("Heatmap of Health Metric Correlations")
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    st.pyplot(plt)
    st.markdown("The heat map above shows the categories that ended up having the highest correlations to calories burned (lightly active minutes, fairly active minutes, and sedintary minutes were all lower than .25)... From the heatmap we can see that TotalDistance and VeryActiveMinutes have very similiar correlations to calories. Total steps is right below both categories. Thus, it is important that we build both simple linear and multivariate regression models to see what health metric is truly the best fit.")

    st.markdown("***")

with st.expander("Model Building"):
    st.markdown("Model Building:")
    st.markdown("With the correlations created, we were then able to plug our Fitbit data into simple linear regressions as well as multivariate regressions. Let’s look at each linear model and some of the key indicators of whether it’s a good fit for predicting calories burned.")

    #Regression 1:
    st.markdown("Step Count to Calories Simple Linear Regression:")

    y = data['Calories']
    x1 = data['TotalSteps']
    x = sm.add_constant(x1)
    results = sm.OLS(y,x).fit()
    st.text(results.summary())

    st.markdown("Anaylsis:")
    st.markdown("R-Square Value -> A value of 0.3 suggests that the model captures some of the underlying relationships between step counts and calories burned, but there is still a significant amount of unexplained variability. Other factors or variables not included in the model might be influencing the remaining 70% of the variability.")
    st.markdown("P-Value -> A value of 0.00 suggests that step count is statistically significant and we can reject the null hypothesis of Step Count having no effect on Calories burned ")
    st.markdown("Coefficent -> According to our model, we can assume that for every step taken a person will burn .0826 calories. Multiply the coefficent by the total stepcount and add the y-intercept of 1675 (this would be the amount of calories you burn without taking any steps) and you can create the line of best fit which is seen below.")

    fig, ax = plt.subplots()
    ax.scatter(data["TotalSteps"], data["Calories"])
    yhat = 0.0826 * data["TotalSteps"] + 1675
    ax.plot(data["TotalSteps"], yhat, lw=4, c='orange', label='regression line')
    ax.set_title("Line of Best Fit Daily Step Count to Calories", fontsize=20)
    ax.set_xlabel('Total Steps', fontsize=20)
    ax.set_ylabel('Calories', fontsize=20)
    st.pyplot(fig)

    #Regression 2:
    st.markdown("Total Distance to Calories Simple Linear Regression:")
    y = data['Calories']
    x1 = data['TotalDistance']
    x = sm.add_constant(x1)
    results = sm.OLS(y,x).fit()
    st.text(results.summary())

    st.markdown("Anaylsis:")
    st.markdown("R-Square Value -> With an .08 increase in r-sqaure value, we can see that Total Distance is a slighlty better indicator of calories burned as it describes 38% of the variability in Calories Burned. However, there is still a significant amount of unexplained variability. Other factors or variables not included in the model might be influencing the remaining 62% of the variability.")
    st.markdown("P-Value -> A value of 0.00 suggests that total distance is statistically significant and we can reject the null hypothesis of total distance walked having no effect on Calories Burned.")
    st.markdown("Coefficent -> According to our model, we can assume that for every miled walked a person will burn 189 calories. Multiply the coefficent by the total stepcount and add the y-intercept of 1660 (this would be the amount of calories you burn without taking any steps) creates the line of best fit which is seen below.")

    fig, ax = plt.subplots()
    ax.scatter(data["TotalDistance"], data["Calories"])
    yhat = 189.13 * data["TotalDistance"] + 1660
    ax.plot(data["TotalDistance"], yhat, lw=4, c='orange', label='regression line')
    ax.set_title("Line of Best Fit Daily Total Distance to Calories", fontsize=20)
    ax.set_xlabel('Total Distance', fontsize=20)
    ax.set_ylabel('Calories', fontsize=20)
    st.pyplot(fig)

    #Regression 3:
    st.markdown("Very Active Minutes to Calories Simple Linear Regression:")
    y = data['Calories']
    x1 = data['VeryActiveMinutes']
    x = sm.add_constant(x1)
    results = sm.OLS(y,x).fit()
    st.text(results.summary())

    st.markdown("Anaylsis:")
    st.markdown("R-Square Value -> With less than .01 in difference between Very Active Minutes and Total Distance, we can see that Very Active Minutes is a slighlty better indicator of calories burned as it describes 37% of the variability in Calories Burned. However, there is still a significant amount of unexplained variability. Other factors or variables not included in the model might be influencing the remaining 63% of the variability.")
    st.markdown("P-Value -> A value of 0.00 suggests that total distance is statistically significant and we can reject the null hypothesis of Daily Very Active Minutes having no effect on Calories Burned.")
    st.markdown("Coefficent -> According to our model, we can assume that for every Very Active Minute a person will burn 12.5 calories. Multiply the coefficent by the total stepcount and add the y-intercept of 2083 (this would be the amount of calories you burn without taking any steps) creates the line of best fit which is seen below.")

    fig, ax = plt.subplots()
    ax.scatter(data["VeryActiveMinutes"], data["Calories"])
    yhat = 12.5631 * data["VeryActiveMinutes"] + 2083
    ax.plot(data["VeryActiveMinutes"], yhat, lw=4, c='orange', label='regression line')
    ax.set_title("Line of Best Fit Daily Very Active Minutes to Calories", fontsize=20)
    ax.set_xlabel('Very Active Minutes', fontsize=20)
    ax.set_ylabel('Calories', fontsize=20)
    st.pyplot(fig)

    #Regression 4:
    st.markdown("Step Count, Total Distance & Very Active Minutes to Calories Multivariate Regression:")
    y = data['Calories']
    x1 = data[['TotalDistance','VeryActiveMinutes','TotalSteps']]
    x = sm.add_constant(x1)
    results = sm.OLS(y,x).fit()
    st.text(results.summary())

    st.markdown("Anaylsis:")
    st.markdown("R-Square Value -> Typically when adding more indpendent variables, a model should see an increase in R-sqaured number. In our model's case, we can see that was a .24 increase in explanation of variabilty of calories burned. However, we need to look at adjusted r squared to see if this is actually significant.")
    st.markdown("Adjusted R-Squared -> Adjusted R-Squared only drops by .002, meaning that this model is a good fit and adding multiple independent variables will help us have a better prediction of calories burned.")
    st.markdown("P-Value -> A value of 0.00 for all independent variables suggests that all variables are statistically significant and we can reject the null hypothesis of Fitness Metrics having no effect on Calories Burned can be rejected.")
    st.markdown("Coefficent of StepCount -> This is where things get a little weird. The coefficient value for step count is negative... Meaning for every step a person will unburn a calorie? This is likely due to multicollinearity between Step Count and Total Distance, or in other words both of these variable have similiar effects on calories burned. Although this is strange, total distance adjusts for the negative step count value by having a higher value.")
    st.markdown("Coefficent of TotalDistance -> For every mile walked, a person will burn 585 calories.")
    st.markdown("Coefficent of VeryActiveMinutes -> For every very active minute, a person will burn 7.5 calories.")
    st.markdown("Conclusion -> All in all, the coefficient numbers look a little funky, but this model ends up being a better fit or predictor because it has more data. In order to get rid of multicollinearity, I would build two multivariate models with step count and very active minutes (.41 adjusted r-squared) + total distance and very active minutes (.44 adjusted r-squared).")
    st.markdown("***")

with st.expander("My Data Vs. The Models"):
    st.markdown("Me vs. the Model:")
    st.markdown("In order to take this report one step further, I decided to take my own Fitbit data and see whether it could be a good indicator of my own weight loss journey via using statsmodelsapi to predict calories burned. I took my stats from January of 2023 through the end of July 2023 and predicted calories burned. I then utilized this data to see if it could be an accurate indicator of my weight.")
    st.markdown("First Prediction: Using Step Count to Predict Calories Burned:")
    if st.checkbox('Show raw data 1'):
        st.subheader('Step Count Prediction Data')
        st.dataframe(stepcount_prediction_df)
    st.markdown("Using the step count linear regression from the fitbit data frame, I was able to use pandas methods to predict my calories burned based upon my daily stepcount. Unfortunately, I saw that the prediction data was quite off. After doing some digging, I determined that the main reason for the difference in calories burned between the predictor and the model was because it was using too low of a Resting Metabolic Rate or starting point for the y-intercept (1675). In terms of body type, I am an outlier. I'm 6' 3'', almost 200 pounds at the start of the year, and only 26. With that information, I plugged my stats into a RMR Calculator that can be found at the link below and saw a 600+ calorie difference in the Models RMR and my Actual RMR. So I adjusted the RMR in the model and the prediction was surprisingly accurate.")
    st.markdown("Below you will see the predictions for my total calories burned over the course of January through July based upon the Fitbit data total steps linear regression. All in all, the adjusted prediction is more accurate then I anticipated accounting for 91% of the calories I actually burned over the course of the time period.")
    # Quick by the numbers...
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("OLS Prediction", "622332")
    col2.metric("Adjusted Prediction", "751652")
    col3.metric("True Calories Burned", "823921")
    col4.metric("Adj. Prediction Accuracy", "91%")

    # Total Calorie Predictor 
    x_values = ["Calories Burned Prediction", "Adjusted Calories Burned", "True Calories Burned"]
    data = [622332, 751652, 823921]
    df = pd.DataFrame({"Values": x_values, "Calories": data})

    sns.set(style="whitegrid")

    plt.figure(figsize=(8, 5))
    sns.barplot(x="Values", y="Calories", data=df, palette="colorblind")
    plt.xlabel("Values")
    plt.ylabel("Calories")
    plt.title("Total Calories Burned Using Step Count Predictors")
    st.pyplot(plt)


    st.markdown("Second Prediction: Using Total Distance to Predict Calories Burned:")
    if st.checkbox('Show raw data 2'):
        st.subheader('Total Distance Prediction Data')
        st.dataframe(totaldistance_prediction_df)
    st.markdown("Using the total distance linear regression from the fitbit data frame, I was able to use pandas methods to predict my calories burned based upon my daily stepcount. Unfortunately, I saw that the prediction data was quite off. After doing some digging, I determined that the main reason for the difference in calories burned between the predictor and the model was because it was using too low of a Resting Metabolic Rate or starting point for the y-intercept (1675). In terms of body type, I am an outlier. I'm 6' 3'', almost 200 pounds at the start of the year, and only 26. With that information, I plugged my stats into a RMR Calculator that can be found at the link below and saw a 600+ calorie difference in the Models RMR and my Actual RMR. So I adjusted the RMR in the model and the prediction was surprisingly accurate.")
    st.markdown("Below you will see the predictions for my total calories burned over the course of January through July based upon the Fitbit data total distance linear regression. All in all, the adjusted prediction is even more accurate then total step count predictions accounting for 96% of the calories I actually burned over the course of the time period.")
    # Quick by the numbers...
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("OLS Prediction", "656633")
    col2.metric("Adjusted Prediction", "789133")
    col3.metric("True Calories Burned", "823921")
    col4.metric("Adj. Prediction Accuracy", "96%")

    # Total Calorie Predictor 
    x_values = ["Calories Burned Prediction", "Adjusted Calories Burned", "True Calories Burned"]
    data = [656633, 789133, 823921]
    df = pd.DataFrame({"Values": x_values, "Calories": data})

    sns.set(style="whitegrid")

    plt.figure(figsize=(8, 5))
    sns.barplot(x="Values", y="Calories", data=df, palette="colorblind")
    plt.xlabel("Values")
    plt.ylabel("Calories")
    plt.title("Total Calories Burned Using Total Distance Predictors")
    st.pyplot(plt)

    st.markdown("Other things to note:")
    st.markdown("I also utilized very active minutes and a multivariate regression to predict my calories burned. Both were less accurate due to a couple of reasons. For active minutes, Fitbit has changed its algorithms since 2016, so I'm not sure if the data I inserted is the same as the data recorded by the participants in the 2016 study. This makes that data irrelevant. Thus, the multivariate models are also mute because they utilize very active minutes. To see what the prediction looked like, feel free to click the checkboxes below:")
    if st.checkbox('Show raw data 3'):
        st.subheader('Very Active Minutes Prediction Data')
        st.dataframe(activeminutes_prediction_df)
    if st.checkbox('Show raw data 4'):
        st.subheader('Multivariate Prediction Data')
        st.dataframe(multivariate_prediction_df)
    st.markdown("***")

with st.expander("Conclusion"):
    st.markdown("Conclusion")
    st.markdown("Is step count a good indicator of calories burned and thus weight loss?")
    st.markdown("Based off of my analysis, I would say that it is a good baseline statistic to judge physical fitness. While it is correlated with calories burned, they are no where near a perfect match or indicator of one another. Yes, if you increase your step count you should see an increase in calories burned and thus a loss in weight. However, there are always some inaccuracies with predictive models, so adjusting my step count won't have a perfect algorithm or indication of weight loss. Thus, anyone looking to lose weight should be tracking a variety of metrics (exercise time, total distance, heart rate, sleep, stress levels, calorie intake, and so much more). To get a better baseline of how many steps you need to take to hit your monthly weight loss goal, feel free to check out my step count weight loss predictor that utilizes the models above to calculate your weight loss!")
    st.markdown("***")