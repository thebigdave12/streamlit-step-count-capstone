import streamlit as st;
import pandas as pd;
import numpy as np;
import seaborn as sns;
import matplotlib.pyplot as plt 

st.title('Does Step Count Really Count???')
st.divider()

st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Fitbit_logo.svg/2560px-Fitbit_logo.svg.png')
st.divider()

st.header('About the Project:')

st.markdown('For the last 8 years, I’ve been an avid Fitbit user. I’ve watched the wearable go from a simple pedometer (step-counter) style tracker to being able to read body temperature, rate your sleep, control anxiety, and tell you how hard you should work out that day when you wake up. Despite all the new features, Fitbit has stayed true to utilizing step count as its predominant metric. With every user having a built-in goal to get 10,000 steps a day, I’ve always wondered if step count is actually a good metric to measure my health??? (of course, “health” is a difficult concept to attribute to one specific metric, but in my study, we’ll be looking at step count’s effect on calories burned/weight loss)')
st.markdown('So in order to answer this question, I’ve utilized a data set of 32 Fitbit users who have actively used their Fitbit over the course of a month (940 data samples to be exact). With their information, we should be able to perform accurate linear and multivariate regressions that can predict calories burned and weight loss based upon step count and other independent variables. As a cherry on top, we’ll use those models to predict my own weight loss over the last 6 months… As an Active Fitbitter, you know I’ve been on top of my step count since the start of the year. Will Fitbit’s bet on step count hold up?!? Check out the report and predictors to get a better idea!')

