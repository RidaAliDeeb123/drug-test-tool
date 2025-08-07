import streamlit as st
from fda_data import FDADrugData
from data_ingestor import DataIngestor
from data_preparation import DrugResponseDataPreparer
from models import DrugResponsePredictor
import pandas as pd
import plotly.express as px

# Configure Streamlit
st.set_page_config(page_title="Drug Risk Assessment Tool", layout="wide")

def main():
    st.title("🚀 Drug Risk Assessment Tool")
    
    # Initialize components
    fda_data = FDADrugData()
    ingestor = DataIngestor()
    preparer = DrugResponseDataPreparer()
    predictor = DrugResponsePredictor()
    
    # Sidebar controls
    st.sidebar.header("Data Source")
    source_type = st.sidebar.selectbox("Select data source", ["CSV", "EHR", "Manual"])
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["FDA Data", "Data Analysis", "Predictions"])
    
    with tab1:
        drug_name = st.text_input("Enter drug name")
        if st.button("Search FDA Database"):
            with st.spinner("Fetching data..."):
                results = fda_data.search_drug(drug_name)
                if results:
                    profile = fda_data.create_drug_profile(drug_name)
                    st.json(profile)
                else:
                    st.error("No results found")
    
    with tab2:
        uploaded_file = st.file_uploader("Upload data file")
        if uploaded_file:
            df = ingestor.load_medical_data("csv", uploaded_file)
            st.dataframe(df.head())
            
            # Bias analysis
            bias_report = ingestor.detect_bias(df)
            st.plotly_chart(px.pie(
                values=[bias_report['male_ratio'], bias_report['female_ratio']],
                names=["Male", "Female"],
                title="Gender Distribution"
            ))
    
    with tab3:
        st.header("Drug Response Predictions")
        if st.button("Train Models"):
            with st.spinner("Training models..."):
                synthetic_data = preparer.create_synthetic_data()
                prepared_data = preparer.prepare_training_data(synthetic_data)
                predictor.train_models(prepared_data['X_train'], prepared_data['y_train'])
                st.success("Models trained successfully!")

if __name__ == "__main__":
    main()