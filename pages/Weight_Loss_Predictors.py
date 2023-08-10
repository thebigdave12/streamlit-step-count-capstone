import streamlit as st;

stepcount_coefficent = 0.0826
st.title('Weight Loss Predictor:')

st.markdown("About the Predictor")
st.markdown("The following predictor utilizes a linear regression model from a data set of over 30 particpants that tracked their health metrics over the course of 1 month. In order to get the most accurate data possible it is essential to determine your Restint Metabolic Rate or the number of calories you burn if you were to take 0 steps. To determine that number, please follow the link below (if you prefer not to enter your info on a secondary site the avg RMR for a male is 1800 and for a female is 1500): ")
st.markdown("https://www.omnicalculator.com/health/rmr")

rmr = st.number_input('Insert your RMR')
st.write('Your RMR is ', rmr)

daily_calorie_intake = st.number_input('Insert Daily Calorie Intake')
st.write('Calorie Intake = ', daily_calorie_intake)

daily_step_count = st.slider('Daily Step Count', 0, 30000, 10000, step=1000)
st.write('Daily Step Count Goal:', daily_step_count)

if st.button('Predict Weight Loss'):
    avg_calories_burned = rmr + (daily_step_count * stepcount_coefficent)
    daily_weight_loss = avg_calories_burned - daily_calorie_intake
    weight_loss_over_30_days = ((30 * daily_weight_loss) / 3500)
    if weight_loss_over_30_days < 0:
        weight_loss_over_30_days = weight_loss_over_30_days * -1
        weight_loss_over_30_days = round(weight_loss_over_30_days, 1)
        st.write(f'You are projected to gain {weight_loss_over_30_days} lbs over the next month...' )
        st.write("Try adjusting your calorie intake or daily step count for better results")
    else:
        weight_loss_over_30_days = round(weight_loss_over_30_days, 1)
        st.write(f'Congrats you are projected to lose {weight_loss_over_30_days} lbs over the next month!' )
        st.balloons()
