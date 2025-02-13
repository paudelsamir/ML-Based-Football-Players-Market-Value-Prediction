import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# Page configuration
st.set_page_config(page_title="Player Value Predictor", layout="wide")
st.title("⚽ Football Player Market Value Predictor")
# Load resources
@st.cache_resource
def load_resources():
    model = joblib.load('model.pkl')
    df = pd.read_pickle('df.pkl')
    with open('team_target_encoding.json', 'r') as f:
        team_encoding = json.load(f)
    return model, df, team_encoding

model, df, team_encoding = load_resources()


# Input form
with st.form("player_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Basic Information")
        position_category = st.selectbox("Position Category", 
                                       ['Forward', 'Midfielder', 'Defender', 'Goalkeeper'], index=1, key='position_category')
        submit = st.form_submit_button("Confirm Position Category")
        age = st.number_input("Age", min_value=16, max_value=45, value=30)
        team = st.selectbox("Team", list(team_encoding.keys()), index=list(team_encoding.keys()).index('Manchester City'))
        foot = st.radio("Preferred Foot", ["Right", "Left"], index=0)
        wage = st.slider("Wage (€ per week)", min_value=500, max_value=440_000, value=270_000)
        years_left = st.number_input("Years Left on Contract", min_value=0, max_value=10, value=3)
        
    with col2:
        st.subheader("Physical Attributes")
        height = st.number_input("Height (cm)", min_value=150, max_value=220, value=181)
        weight = st.number_input("Weight (kg)", min_value=50, max_value=120, value=70)
        acceleration = st.slider("Acceleration", 1, 100, 78)
        sprint_speed = st.slider("Sprint Speed", 1, 100, 76)
        
    with col3:
        st.subheader("Additional Info")
        international_reputation = st.slider("International Reputation", 1, 5, 5)
        on_loan = st.radio("On Loan", ["No", "Yes"], index=0)
        agility = st.slider("Agility", 1, 100, 82)
        balance = st.slider("Balance", 1, 100, 80)
        stamina = st.slider("Stamina", 1, 100, 90)
        strength = st.slider("Strength", 1, 100, 74)
    
    # Position-specific skills
    st.subheader("Position-specific Skills")
    skills = {}
    position_cols = st.columns(4)
    
    if position_category == 'Forward':
        with position_cols[0]:
            skills['crossing'] = st.slider('Crossing', 1, 100, 75)
            skills['finishing'] = st.slider('Finishing', 1, 100, 82)
        with position_cols[1]:
            skills['dribbling'] = st.slider('Dribbling', 1, 100, 88)
            skills['ball_control'] = st.slider('Ball Control', 1, 100, 85)
        with position_cols[2]:
            skills['volleys'] = st.slider('Volleys', 1, 100, 78)
    elif position_category == 'Midfielder':
        with position_cols[0]:
            skills['vision'] = st.slider('Vision', 1, 100, 94)
            skills['long_passing'] = st.slider('Long Passing', 1, 100, 93)
        with position_cols[1]:
            skills['short_passing'] = st.slider('Short Passing', 1, 100, 92)
            skills['composure'] = st.slider('Composure', 1, 100, 88)
    elif position_category == 'Defender':
        with position_cols[0]:
            skills['defensive_awareness'] = st.slider('Defensive Awareness', 1, 100, 82)
            skills['standing_tackle'] = st.slider('Standing Tackle', 1, 100, 85)
        with position_cols[1]:
            skills['interceptions'] = st.slider('Interceptions', 1, 100, 83)
            skills['aggression'] = st.slider('Aggression', 1, 100, 75)
    else:  # Goalkeeper
        with position_cols[0]:
            skills['gk_diving'] = st.slider('GK Diving', 1, 100, 80)
            skills['gk_handling'] = st.slider('GK Handling', 1, 100, 78)
        with position_cols[1]:
            skills['gk_reflexes'] = st.slider('GK Reflexes', 1, 100, 85)
    
    submitted = st.form_submit_button("Predict Market Value")

# Prediction and results
if submitted:
    # Calculate position score
    position_score = np.mean(list(skills.values())) if skills else 0
    
    # Create feature array
    features = {
        'Age': age,
        'foot': 1 if foot == "Right" else 0,
        'Wage': np.log1p(wage),
        'Height_cm': height,
        'Weight_kg': weight,
        'Acceleration': acceleration,
        'Sprint speed': sprint_speed,
        'Agility': agility,
        'Balance': balance,
        'Stamina': stamina,
        'Strength': strength,
        'International reputation': international_reputation,
        'On Loan': 1 if on_loan == "Yes" else 0,
        'Team_encoded': team_encoding[team],
        'Years left': years_left,
        'Forward Score': position_score if position_category == 'Forward' else 0,
        'Midfielder Score': position_score if position_category == 'Midfielder' else 0,
        'Defender Score': position_score if position_category == 'Defender' else 0,
        'Goalkeeper Score': position_score if position_category == 'Goalkeeper' else 0,
        'Position Category_Defender': 1 if position_category == 'Defender' else 0,
        'Position Category_Forward': 1 if position_category == 'Forward' else 0,
        'Position Category_Goalkeeper': 1 if position_category == 'Goalkeeper' else 0,
        'Position Category_Midfielder': 1 if position_category == 'Midfielder' else 0,
    }
    
    # Convert to DataFrame and predict
    input_df = pd.DataFrame([features])
    log_prediction = model.predict(input_df)[0]
    prediction = np.exp(log_prediction)  # Apply exponential to reverse log transform
    
    # Display results
    st.success(f"Predicted Market Value: €{prediction:,.2f} million")
    
