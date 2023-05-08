
##Imporing the required libraries 
## Required Libaries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
plt.style.use('ggplot')

## Calling the important functions

##==================== Function Section: ==================

def welcome():
    hello = st.title('👋 Welcome to Auto Insurance Fraud Detection : ')
    return hello

def dataset(nrows):
    data = pd.read_csv('./DataSet/insurance_claims.csv')
    return data.head(nrows)

def dataStats(df):
    stats= df.describe().transpose()
    return stats

def dataCleaning(df):
    df.replace('?', np.nan, inplace = True) # replacing '?' with 'nan' value
    #df.isna().sum() / len(df) * 100
    return 200
    
def agedistgraph(df):
    #Age Distribution
    plt.hist(df['age'], bins=20)
    plt.xlabel('Age')
    plt.ylabel('Count')
    return plt.show()
    
    
def visualizeCategorical():
    return 200
    

##==================== Main Application Function ==========

## start the streamlit app 
def app():
    ##starting with a header 
    welcome()
    ##Here we are going to showcase our dataset:
    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
    # Load 10,000 rows of data into the dataframe.
    data = dataset(10)
    # Notify the reader that the data was successfully loaded.
    data_load_state.text('Loading data...done!')
    st.subheader('1- Step 1: 📊 Raw data')
    st.write(data)
    data_load_state.text("Done!")
    st.subheader('2- Step 2: 📈 Data Statistics:')
    stats = dataStats(dataset(1000))
    st.write(stats)
             
    st.subheader('3- Step 3: 📉 Data Visualization:')
    st.write("Here we are going to showcase the EDA process through visualizatio plots:")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(agedistgraph(dataset(1000)))

    
    return 200


## Executing the app 
if __name__ == "__main__":
    app()
