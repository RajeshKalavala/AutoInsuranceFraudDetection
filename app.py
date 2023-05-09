
##Imporing the required libraries 
## Required Libaries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

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

## Create a function that returns the list of all the most relevant columns as per responsiveness to a model treshhold 
def columns_selection(dataframe):
    '''
    Selecting only columns that are having >10 unique values 
    '''
    ## Preparing the dataframe 
    ## converting the nunique output to a dataframe 
    output = pd.DataFrame(dataframe.nunique())
    output.reset_index(inplace=True)
    output.columns = ["features","counts"]
    ## making a list of relevant columns 
    col_list = []
    final_columns = []
    ## Process section: Making an iteratable list
    mycounts = output["counts"].values.tolist()
    for value in mycounts:
        if (value < 10):
            col_list.append(output[output.counts.isin([value,])]) 
    ## Output Section
    return col_list

def scale_and_encode(df):
    # Split columns into numerical and categorical
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Standardize numerical columns
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=cat_cols,drop_first = True)
    
    return df
    
def fit_and_score(models, X_train, X_test, y_train, y_test):
    np.random.seed(0)
    
    model_scores = {}
    
    for name, model in models.items():
        model.fit(X_train,y_train)
        model_scores[name] = model.score(X_test,y_test)

    model_scores = pd.DataFrame(model_scores,index=['Score']).transpose()
    model_scores = model_scores.sort_values('Score')
    
    
    return model_scores
    
##==================== Main Application Function ==========

## start the streamlit app 
def app():
    ##starting with a header 
    welcome()
    ##Here we are going to showcase our dataset:
    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
    # Load 10,000 rows of data into the dataframe.
    data = dataset(1000)
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
    st.pyplot(agedistgraph(dataset(10)))
    
    st.subheader('4- step4: BOX plot:')
    st.write("policy annual premium by incident severity:")
    df = data
    df.info()
    fig = plt.boxplot([df[df["incident_severity"]=="Minor Damage"]["policy_annual_premium"],
             df[df["incident_severity"]=="Major Damage"]["policy_annual_premium"],
             df[df["incident_severity"]=="Total Loss"]["policy_annual_premium"],
             df[df["incident_severity"]=="Trivial Damage"]["policy_annual_premium"]],
            labels=["Minor Damage", "Major Damage", "Total Loss", "Trivial Damage"])
    plt.title("Policy Annual Premium by Incident Severity")
    plt.xlabel("Incident Severity")
    plt.ylabel("Policy Annual Premium")
    st.write(plt.show())
    st.pyplot()
    
    ## STEP 5
    st.subheader('5- step5: BOX plot:')
    st.write("----:")
    plt.figure(figsize=(8,6))
    df['incident_type'].value_counts().plot(kind='bar')
    plt.title('Count of Claims by Incident Type')
    plt.xlabel('Incident Type')
    plt.ylabel('Count of Claims')
    st.write(plt.show())
    st.pyplot()
    
    ## STEP 6
    st.subheader('6- step6: Missing Value Adjustment:')
    st.write("Missing Value Adjustment:")
    ### Calculating the corresponding means per collision_type, property_damage, and police_report_available
    #mean_of_collision_type = df['collision_type'].mean()
    #mean_of_property_damage = df['property_damage'].mean()
    #mean_of_police_report_available = df['police_report_available'].mean()'
    mean_of_property_claim = df['property_claim'].mean()
    st.write(mean_of_property_claim)
    
    ## STEP 7
    st.subheader('7- step7: Finding Feature Correlation:')
    st.write("Finding Feature Correlation:")
    df['property_claim'] = df['property_claim'].fillna(value=mean_of_property_claim)

    ## since we haven't encoded the object types into numerical, we opt for mode as a solution to rectify the missing values 
    df['collision_type'] = df['collision_type'].fillna(df['collision_type'].mode()[0])
    df['property_damage'] = df['property_damage'].fillna(df['property_damage'].mode()[0])
    df['police_report_available'] = df['police_report_available'].fillna(df['police_report_available'].mode()[0])

    df.isna().sum()


    st.write(df['property_claim'])


    st.subheader('8- step8: Heat Plot:')

    # heatmap
    st.write("Heat Plot:")

    plt.figure(figsize = (18, 12))

    corr = df.corr()
    st.write(sns.heatmap(data = corr, annot = True, fmt = '.2g', linewidth = 1))
    st.pyplot()
    

    st.write(df.nunique())
    
    ## previewing the list of relevant columns 
    columns_selection(df)

    # dropping columns which are not necessary for prediction

    to_drop = ['policy_number','policy_bind_date','policy_state','insured_zip','incident_location','incident_date',
        'insured_hobbies','auto_make','auto_model','auto_year', '_c39']

    df.drop(to_drop, inplace = True, axis = 1)
    
    ### Using Plotly Dash to locate the claims per map
    
    ## Plotting the map
    st.subheader('9- step9: Using Plotly Dash to locate the claims per map:')

    # heatmap
    st.write("plotting the map:")
    
    geojson = px.data.election_geojson()
    fig = px.choropleth(
            df, geojson=geojson, color='incident_state',
            locations='incident_city', featureidkey="properties.incident_state",
            projection="mercator", range_color=[0, 6500])

    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    # return the figure
    st.write(fig.show())
    st.pyplot()

    msg = f'shape of dataset: {df.shape}'
    st.write(msg)
    st.write(df.head())

    ### define the technical urge to use the multicollinearity
    st.subheader('10 - Step10: checking for multicollinearity')
    # checking for multicollinearity

    plt.figure(figsize = (18, 12))

    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype = bool))
    
    st.write(sns.heatmap(data = corr, mask = mask, annot = True, fmt = '.2g', linewidth = 1))
    st.pyplot()
    
    
    df.drop(columns = ['age', 'total_claim_amount'], inplace = True, axis = 1)
    st.write(df.head())

    # separating the feature and target columns
   

    X = df.drop('fraud_reported', axis = 1)
    y = df['fraud_reported']

    y = y.map({'Y': 1, 'N': 0})    # Use map() to replace 'Y' with 1 and 'N' with 0

    # Split columns into numerical and categorical
    
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    ### Visualizing numerical columns 

    #st.write("Visualizing numerical columns")

    #df.drop(columns = ['age', 'total_claim_amount'], inplace = True, axis = 1)
    #st.write(df.head())
    
    
    plt.figure(figsize = (25, 20))
    plotnumber = 1

    for col in num_cols:
        if plotnumber <= 24:
            ax = plt.subplot(5, 5, plotnumber)
            sns.distplot(X[col])
            plt.xlabel(col, fontsize = 15)

        plotnumber += 1

    st.write(plt.tight_layout())
    st.pyplot()

    # Outliers Detection
    st.write("Outliers Detection")
    plt.figure(figsize = (20, 15))
    plotnumber = 1

    for col in num_cols:
        if plotnumber <= 24:
            ax = plt.subplot(5, 5, plotnumber)
            sns.boxplot(X[col])
            plt.xlabel(col, fontsize = 15)

        plotnumber += 1
    st.write(plt.tight_layout())
    st.pyplot()
    
    st.write("Converting categoirical data into quantitive data")


    X = scale_and_encode(X)
    X.head()

    #Data Splitting for ML deployment

    # splitting data into training set and test set

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=42,stratify=y)
    
    models = {'LogisticRegression': LogisticRegression(max_iter=10000),
          'KNeighborsClassifier': KNeighborsClassifier(),
          'SVC': SVC(),
          'GaussianNB':GaussianNB(),
          'DecisionTreeClassifier': DecisionTreeClassifier(),
          'RandomForestClassifier': RandomForestClassifier(),
          'GradientBoostingClassifier': GradientBoostingClassifier(),
          'AdaBoostClassifier': AdaBoostClassifier(),
          'XGBClassifier': XGBClassifier(),
          'CatBoostClassifier':CatBoostClassifier()}
    
   
    model_scores = fit_and_score(models,X_train,X_test,y_train,y_test)
    st.write(model_scores)
        
## Executing the app 
if __name__ == "__main__":
    app()
