import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import smtplib
import ssl
from email.message import EmailMessage
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np

st.set_page_config(page_title="AutoStat-AI++", layout="wide") 
st.title("📊 AutoStat-AI++: Survey Data Cleaner & Analyzer")

# Step 1: File Upload
uploaded_file = st.file_uploader("📤 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Clean numeric columns
    for col in ['Income', 'Year']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=['Income', 'Year'], inplace=True)

    # Step 2: Raw Preview
    st.subheader("📁 Raw Data Preview")
    st.dataframe(df)

    # Step 3: Missing Values
    st.subheader("🚫 Missing Values Check")
    st.write(df.isnull().sum())

    if st.button("Remove missing values"):
        df = df.dropna()
        st.success("✅ Removed missing rows")

    # Step 4: Data Types
    st.subheader("🧬 Data Types")
    st.write(df.dtypes)

    # Step 5: Duplicates
    dup_count = df.duplicated().sum()
    st.write(f"Found {dup_count} duplicate rows")

    if st.button("Remove duplicate rows"):
        df = df.drop_duplicates()
        st.success("✅ Removed duplicate rows")

    # Step 6: Cleaned Data
    st.subheader("✅ Cleaned Data Preview")
    st.dataframe(df)

    # ✅ CSV DOWNLOAD SECTION
    st.subheader("📥 Download Cleaned Data")

    @st.cache_data
    def convert_df_to_csv(dataframe):
        return dataframe.to_csv(index=False).encode('utf-8')

    csv_data = convert_df_to_csv(df)

    st.download_button(
        label="📥 Download CSV",
        data=csv_data,
        file_name='cleaned_survey_data.csv',
        mime='text/csv'
    )

    # 📧 EMAIL SECTION
    def send_email_with_attachment(receiver, subject, body, attachment_data, filename):
        sender = st.secrets["email"]["sender"]
        password = st.secrets["email"]["password"]

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = receiver
        msg.set_content(body)

        msg.add_attachment(attachment_data, maintype="application", subtype="octet-stream", filename=filename)

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender, password)
            server.send_message(msg)

    st.subheader("📧 Email Cleaned Data")
    receiver_email = st.text_input("Enter recipient email address")
    send_button = st.button("📤 Send Email")

    if send_button:
        if receiver_email:
            try:
                send_email_with_attachment(
                    receiver=receiver_email,
                    subject="📊 AutoStat-AI++ Cleaned Survey Data",
                    body="Attached is your cleaned dataset in CSV format.",
                    attachment_data=csv_data,
                    filename="cleaned_survey_data.csv"
                )
                st.success("✅ Email sent successfully to " + receiver_email)
            except Exception as e:
                st.error(f"❌ Failed to send email: {e}")
        else:
            st.warning("⚠️ Please enter a valid email address.")

    # Step 7: Summary Stats
    st.subheader("📊 Estimation and Summary Stats")
    st.write(df.describe())

    # Step 8: Filters
    st.sidebar.header("🔍 Filter Options")
    gender_filter = st.sidebar.selectbox("Select Gender", ["All"] + list(df['Gender'].unique()))
    area_filter = st.sidebar.selectbox("Select Area", ["All"] + list(df['Area'].unique()))

    filtered_df = df.copy()
    if gender_filter != "All":
        filtered_df = filtered_df[filtered_df['Gender'] == gender_filter]
    if area_filter != "All":
        filtered_df = filtered_df[filtered_df['Area'] == area_filter]

    st.write(f"**Filtered Data ({len(filtered_df)} records)**")
    st.dataframe(filtered_df)

    # Step 9: Bar Chart
    st.subheader("📚 Average Income by Education Level")
    income_by_edu = filtered_df.groupby('Education')['Income'].mean().sort_values(ascending=False)
    st.bar_chart(income_by_edu)

    # Step 10: Line Chart
    if 'Year' in df.columns:
        st.subheader("📈 Income Trend by Year")
        trend_data = filtered_df.groupby('Year')['Income'].mean()
        st.line_chart(trend_data)

    # Step 11: Pie Chart
    st.subheader("🎯 Gender Distribution")
    gender_count = filtered_df['Gender'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(gender_count, labels=gender_count.index, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)

    # Step 12: ML Training
    st.header("🤖 Machine Learning Model Training")
    all_cols = list(df.columns)
    input_features = st.multiselect("Select input features", all_cols, default=["Education", "Gender", "Year"])
    target_col = st.selectbox("Select target column", [col for col in all_cols if col not in input_features])

    if st.button("🚀 Train & Save Model"):
        df_model = df[input_features + [target_col]].copy()
        for col in df_model.columns:
            if df_model[col].dtype == 'object':
                le = LabelEncoder()
                df_model[col] = le.fit_transform(df_model[col])

        X = df_model[input_features]
        y = df_model[target_col]
        model = LinearRegression()
        model.fit(X, y)
        joblib.dump(model, "model.pkl")
        joblib.dump(input_features, "features.pkl")
        st.success("✅ Model trained and saved successfully!")

    # Step 13: Prediction
    if os.path.exists("model.pkl") and os.path.exists("features.pkl"):
        st.header("🔮 Prediction Module")
        model = joblib.load("model.pkl")
        features = joblib.load("features.pkl")
        input_data = {}

        for col in features:
            if df[col].dtype == 'object':
                input_data[col] = st.selectbox(f"{col}", df[col].unique(), key=col)
            else:
                input_data[col] = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), key=col)

        input_df = pd.DataFrame([input_data])

        for col in input_df.columns:
            if df[col].dtype == 'object':
                le = LabelEncoder()
                le.fit(df[col])
                input_df[col] = le.transform(input_df[col])

        input_df = input_df.reindex(columns=features, fill_value=0)
        pred = model.predict(input_df)[0]
        st.success(f"🎯 Predicted {target_col}: {round(pred, 2)}")

else:
    st.info("📂 Please upload a CSV file to begin.")