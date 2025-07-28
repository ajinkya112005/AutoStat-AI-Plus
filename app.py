import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import altair as alt
import smtplib
import ssl
from email.message import EmailMessage
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
from fpdf import FPDF
import tempfile
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

st.set_page_config(page_title="AutoStat-AI++", layout="wide")

st.title("📊 AutoStat-AI++: Survey Data Cleaner & Analyzer")

admin_view = st.sidebar.checkbox("👮 Admin Dashboard")

uploaded_file = st.file_uploader("📄 Upload your CSV/Excel file", type=["csv", "xlsx"])

st.subheader("📤 Or Enter Google Sheets URL")
gsheet_url = st.text_input("Paste Google Sheets URL (Public/Shared)")

if uploaded_file is not None or gsheet_url:
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    elif gsheet_url:
        try:
            sheet_id = gsheet_url.split("/d/")[1].split("/")[0]
            gsheet_csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
            df = pd.read_csv(gsheet_csv_url)
            st.success("✅ Loaded Google Sheet successfully")
        except Exception as e:
            st.error(f"❌ Failed to read Google Sheet: {e}")
            st.stop()

    for col in ['Income', 'Year']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['Income', 'Year'], inplace=True)

    st.subheader("📁 Raw Data Preview")
    st.dataframe(df, use_container_width=True)

    st.subheader("📈 Interactive Dashboard - Plotly")
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    if numeric_cols and categorical_cols:
        x_col = st.selectbox("Choose X-axis (Categorical)", categorical_cols)
        y_col = st.selectbox("Choose Y-axis (Numeric)", numeric_cols)
        fig = px.bar(df, x=x_col, y=y_col, color=x_col, title=f"Bar Chart of {y_col} by {x_col}")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Interactive Dashboard - Altair")
    if numeric_cols and categorical_cols:
        bar_chart = alt.Chart(df).mark_bar().encode(
            x=alt.X(f'{x_col}:N', title=x_col),
            y=alt.Y(f'{y_col}:Q', title=y_col),
            color=x_col
        ).properties(width=700, height=400)
        st.altair_chart(bar_chart, use_container_width=True)

    st.subheader("🧠 Auto Insights - Natural Language Summary")
    if st.button("🪄 Generate Auto Insights"):
        try:
            profile = ProfileReport(df, title="📊 Auto Insights Report", explorative=True)
            st_profile_report(profile)
        except Exception as e:
            st.error(f"❌ Could not generate auto insights: {e}")

    st.subheader("🧬 Data Types")
    st.write(df.dtypes)

    df.drop_duplicates(inplace=True)

    st.subheader("✅ Cleaned Data Preview")
    st.dataframe(df, use_container_width=True)

    @st.cache_data
    def convert_df_to_csv(dataframe):
        return dataframe.to_csv(index=False).encode('utf-8')
    csv_data = convert_df_to_csv(df)
    st.download_button("📅 Download CSV", data=csv_data, file_name='cleaned_survey_data.csv', mime='text/csv')

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

    st.subheader("📚 Average Income by Education Level")
    if 'Education' in df.columns and 'Income' in df.columns:
        income_by_edu = df.groupby('Education')['Income'].mean().sort_values(ascending=False)
        st.bar_chart(income_by_edu)

    if 'Year' in df.columns and 'Income' in df.columns:
        st.subheader("📈 Income Trend by Year")
        trend_data = df.groupby('Year')['Income'].mean()
        st.line_chart(trend_data)

    if 'Gender' in df.columns:
        st.subheader("🎯 Gender Distribution")
        gender_count = df['Gender'].value_counts()
        fig1, ax1 = plt.subplots()
        ax1.pie(gender_count, labels=gender_count.index, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        st.pyplot(fig1)

    st.subheader("🧪 Compare ML Models")
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(n_estimators=100),
        "KNN": KNeighborsRegressor()
    }

    all_cols = list(df.columns)
    input_features = st.multiselect("Select Input Features", options=all_cols, default=["Education", "Gender", "Year"])
    target_col = st.selectbox("Select Target Column", options=[col for col in all_cols if col not in input_features], index=0)

    best_model = None
    lowest_rmse = float("inf")

    results = []
    if st.button("🏁 Compare and Select Best Model"):
        df_model = df[input_features + [target_col]].copy()
        for col in df_model.columns:
            if df_model[col].dtype == 'object':
                le = LabelEncoder()
                df_model[col] = le.fit_transform(df_model[col])
        X = df_model[input_features]
        y = df_model[target_col]

        model_predictions = []
        for name, model in models.items():
            model.fit(X, y)
            preds = model.predict(X)
            rmse = np.sqrt(mean_squared_error(y, preds))
            results.append((name, rmse))
            model_predictions.append(preds)
            if rmse < lowest_rmse:
                lowest_rmse = rmse
                best_model = model
                best_model_name = name

        # Ensemble Model using Stacking
        meta_X = pd.DataFrame({name: preds for (name, _), preds in zip(results, model_predictions)})
        meta_model = LinearRegression()
        meta_model.fit(meta_X, y)
        meta_preds = meta_model.predict(meta_X)
        meta_rmse = np.sqrt(mean_squared_error(y, meta_preds))
        results.append(("Supermodel (Ensemble)", meta_rmse))

        st.write(pd.DataFrame(results, columns=["Model", "RMSE"]))

        joblib.dump(meta_model, "meta_model.pkl")
        joblib.dump(models, "base_models.pkl")
        joblib.dump(input_features, "features.pkl")
        joblib.dump(target_col, "target.pkl")
        st.success(f"✅ Best model is {best_model_name} with RMSE: {lowest_rmse:.2f} | Ensemble RMSE: {meta_rmse:.2f}")

    # 🤖 AI Chatbot Assistant
st.subheader("🤖 Ask the AI Assistant")
openai_api_key = st.secrets["openai_api_key"] if "openai_api_key" in st.secrets else ""
if openai_api_key:
    llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
    memory = ConversationBufferMemory()
    conversation = ConversationChain(llm=llm, memory=memory)

    user_input = st.text_input("Ask something about your data, forecast, or models:")
    if user_input:
        response = conversation.run(user_input)
        st.success(response)
else:
    st.info("⚠️ Set your OpenAI API key in Streamlit secrets to use the chatbot.")
