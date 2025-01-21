import streamlit as st
import pandas as pd
from pymongo import MongoClient
import altair as alt
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

@st.cache_resource
def connect_to_mongodb():
    client = MongoClient('mongodb://mongodb:27017/')
    db = client['customer_dashboard']
    return db['customer_feedback']
collection = connect_to_mongodb()

@st.cache_data(ttl=10)   #reload data   
def load_data():
    data = list(collection.find()) 
    return pd.DataFrame(data)
df = load_data()

st.title("Customer Insights Dashboard")

st.sidebar.header("Filters")
country_opt=['All']+list(df['Country'].unique())
country = st.sidebar.selectbox("Select Country", country_opt)
loyalty_opt =['All']+ list(df['LoyaltyLevel'].unique())
loyalty=st.sidebar.selectbox("Select loyalty Level",loyalty_opt)

filtered_df =df.copy()
if country!='All':
    filtered_df=df[df['Country'] == country]
if loyalty!='All':
    filtered_df= df[df['LoyaltyLevel']== loyalty]
alt.data_transformers.disable_max_rows()


#Age distribution
st.subheader("Customers Age Distribution")
age_dist = alt.Chart(filtered_df).mark_bar().encode(
    alt.X('Age:Q', bin=alt.Bin(maxbins=30), title='Age Distribution'),
    alt.Y('count()', title='Number of Customers'),
    tooltip=['count()']).properties( width=700, height=500)
st.altair_chart(age_dist, use_container_width=True)

#gender distribution
st.subheader("Customers Gender Distribution")
gend_dist=alt.Chart(filtered_df).mark_arc().encode(
    theta='count():Q',
    color='Gender:N'
    ).properties(width=200, height=200)
st.altair_chart(gend_dist, use_container_width=True)

st.subheader("Satisfaction Score Distribution")
satis_dist=alt.Chart(filtered_df).mark_bar().encode(
    alt.X('SatisfactionScore:Q', bin=alt.Bin(maxbins=30), title='SatisfactionScore Distribution'),
    alt.Y('count()', title='No. of Customers'),
    tooltip=['count()']
    ).properties(width=500, height=400)
st.altair_chart(satis_dist,use_container_width=True)

#Bivariate Analysis
#income x satisfaction score
st.subheader("Income vs Satisfaction Score")
income_satis_chart = alt.Chart(filtered_df).mark_circle(size=60).encode(
    alt.X('Income:Q', title='Income'),
    alt.Y('SatisfactionScore:Q', title='Satisfaction Score'),
    alt.Color('LoyaltyLevel:N', legend=alt.Legend(title="Loyalty Level")),
    tooltip=['Income', 'SatisfactionScore', 'LoyaltyLevel']
).properties( width=700,height=500)
st.altair_chart(income_satis_chart,use_container_width=True)

#Age x purchase history
st.subheader("Age vs Purchase Frequency")
age_purchase_chart = alt.Chart(filtered_df).mark_circle(size=60).encode(
    alt.X('Age:Q', title='Age'),
    alt.Y('PurchaseFrequency:Q', title='Purchase Frequency'),
    alt.Color('Gender:N', legend=alt.Legend(title="Gender")),
    tooltip=['Age', 'PurchaseFrequency', 'Gender']
).properties(width=700,height=500)
st.altair_chart(age_purchase_chart, use_container_width=True)

#gender x loyalty level
st.subheader("Gender vs Loyalty Level")
gender_loyalty_chart = alt.Chart(filtered_df).mark_bar().encode(
    alt.X('Gender:N', title='Gender'),
    alt.Y('count()', title='Number of Customers'),
    alt.Color('LoyaltyLevel:N', legend=alt.Legend(title="Loyalty Level")),
    tooltip=['Gender', 'LoyaltyLevel', 'count()']
).properties(width=700,height=500)
st.altair_chart(gender_loyalty_chart, use_container_width=True)

#product x service quality
st.subheader("Product Quality vs Service Quality")
quality_heatmap = alt.Chart(filtered_df).mark_rect().encode(
    alt.X('ProductQuality:Q', bin=alt.Bin(maxbins=10), title='Product Quality'),
    alt.Y('ServiceQuality:Q', bin=alt.Bin(maxbins=10), title='Service Quality'),
    alt.Color('count()', scale=alt.Scale(scheme='blues'), legend=alt.Legend(title="Number of Customers")),
    tooltip=['ProductQuality', 'ServiceQuality', 'count()']
).properties(width=700,height=500)
st.altair_chart(quality_heatmap, use_container_width=True)

#Categorical Analysis
#Countrywise cust distribution
st.subheader("Countrywise Customer Distribution")
country_dist_chart = alt.Chart(filtered_df).mark_bar().encode(
    alt.X('Country:N',title='Country'),
    alt.Y('count()', title='Number of Customers'),
    alt.Color('Country:N', legend=None),
    tooltip=['Country', 'count()']
).properties(width=700,height=500)
st.altair_chart(country_dist_chart, use_container_width=True)

#Loyality x Country
st.subheader("Loyalty Level Distribution by Country")
loyalty_country_chart = alt.Chart(filtered_df).mark_bar(size=20).encode(
    alt.X('Country:N', title='Country', axis=alt.Axis(labelAngle=-45)),
    alt.Y('count()', title='Number of Customers'),
    alt.Color('LoyaltyLevel:N', legend=alt.Legend(title='Loyalty Level')),
    alt.XOffset('LoyaltyLevel:N'),  # Slight offset to create grouping
    alt.Tooltip(['Country', 'LoyaltyLevel', 'count()'])
).properties(
    width=700,  # Explicitly set width
    height=400,  # Set appropriate height
    title='Loyalty Level Distribution by Country'
).configure_axis(
    labelFontSize=10,
    titleFontSize=12
).configure_legend(
    titleFontSize=12,
    labelFontSize=10
)
st.altair_chart(loyalty_country_chart, use_container_width=False)

#model
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
LABEL_MAPPING = {
    'LABEL_0': 'Low',    # -ve sentiment
    'LABEL_1': 'Medium', # Neutral sentiment
    'LABEL_2': 'High'    # +ve sentiment
}

@st.cache_resource
def load_sentiment_model():
    sentiment_pipeline = pipeline(
        "text-classification",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        device=0 if torch.cuda.is_available() else -1
    )
    return sentiment_pipeline
sentiment_pipeline = load_sentiment_model()

st.title("Customer Feedback Sentiment Analysis")
st.write("""
Enter Customer feedback to get sentiment prediction:  
- **Low (😡 indicates negative feedback )**  
- **Medium (😐 indicates neutral feedback)**  
- **High (😊 indiactes positive feedback)**  
""")
user_input = st.text_area("Enter customer feedback:", placeholder="Type feedback here..")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        with st.spinner("Analyzing sentiment..."):
            try:
                prediction = sentiment_pipeline(user_input) #prediction
                st.write("**🔍 Raw Prediction Output:**", prediction)

                label = prediction[0]['label'] #labelmappinng
                sentiment = LABEL_MAPPING.get(label, "Unknown")
                confidence = float(prediction[0]['score']) * 100
                #results:
                st.write(f"**Predicted Sentiment:** `{sentiment}`")
                st.write(f"**Confidence Score:** `{confidence:.2f}%`")
                sentiment_emoji = {
                    'Low': '😡 Negative',
                    'Medium': '😐 Neutral',
                    'High': '😊 Positive'
                }
                st.write(f"**🎭 Emotion Representation:** {sentiment_emoji.get(sentiment, '🤖 Unknown')}")
            except Exception as e:
                st.error(f"Error during sentiment analysis: {e}")
    else:
        st.warning("Pls enter some feedback to analyze sentiment")