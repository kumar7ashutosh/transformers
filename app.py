import streamlit as st
import pandas as pd
import numpy as np
import neattext.functions as nfx
import joblib

pipeline=joblib.load(open('pipeline_file.pkl','rb'))
st.title("Emotion Detection App")
st.write("Enter text to detect the emotion!")

user_input = st.text_area("Enter your text here:")

if st.button("Detect Emotion"):
    if user_input:
        cleaned_text = nfx.remove_userhandles(user_input)
        cleaned_text = nfx.remove_stopwords(cleaned_text)

        prediction = pipeline.predict([cleaned_text])
        prediction_proba = pipeline.predict_proba([cleaned_text])

        predicted_emotion = prediction[0]

        proba_df = pd.DataFrame(prediction_proba, columns=pipeline.classes_)

        st.subheader("Predicted Emotion:")
        st.success(f"**{predicted_emotion.capitalize()}**")

        st.subheader("Probability Distribution:")
        st.dataframe(proba_df)

        st.bar_chart(proba_df.T) 

    else:
        st.warning("Please enter some text to detect emotion.")