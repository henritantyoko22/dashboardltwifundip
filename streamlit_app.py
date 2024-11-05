#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import numpy as np
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from scipy.stats import linregress
from streamlit_option_menu import option_menu  # Ensure this is installed
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


import pandas as pd
from sklearn.preprocessing import StandardScaler

from constants import CLUSTER_ALGORITHMS, DATASET_NAMES, MAX_PLOTS_PER_ROW
from utils.preparation import get_default_dataset_points, get_cluster_algo_parameters, add_user_data_input_listener, split_list
from utils.modeling import get_cluster_labels
from utils.visualization import plot_figure
from utils.dimension_reduction import apply_standardization, dimensionality_reduction
from data_classes import DatasetName, DimensionReductionAlgo

#######################
# Page configuration
st.set_page_config(
    page_title="Kelulusan Tepat Waktu",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#######################
# CSS styling
st.markdown("""
<style>

[data-testid="block-container"] {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 1rem;
    padding-bottom: 0rem;
    margin-bottom: -7rem;
}

[data-testid="stVerticalBlock"] {
    padding-left: 0rem;
    padding-right: 0rem;
}

[data-testid="stMetric"] {
    background-color: #010351;
    color: #FFFFFF;
    text-align: center;
    padding: 15px 0;
    
}
[data-testid="stMetricValue"] {
    color: #FFFFFF;
    text-align: center;
    
}

[data-testid="stMetricLabel"] {
  display: flex;
  justify-content: center;
  align-items: center;
  color: #FFFFFF;
}

[data-testid="stMetricDeltaIcon-Up"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

[data-testid="stMetricDeltaIcon-Down"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

</style>
""", unsafe_allow_html=True)


#######################
# Load data
df_reshaped = pd.read_csv('data/us-population-2010-2019-reshaped.csv')
# Load dataset
file_path = 'Mahasiswa Lulus Undip.xlsx'
data_mahasiswa = pd.read_excel(file_path, sheet_name='Data Mahasiswa Lulus')

#######################
# Sidebar
with st.sidebar:
    st.title('Dashboard Lulus Tepat Waktu S1 Informatika Universitas Diponegoro')
    selected = option_menu("Main Menu", ["Dashboard Statistic", "Clustering", "Prediki Kelulusan"], 
                        icons=['kanban', 'gear', 'search'],
                        menu_icon="cast", default_index=0)
    


#######################
# Dashboard Main Panel

if selected == 'Dashboard Statistic':
    col = st.columns((2.5, 3.5, 2.5), gap='medium')
    col2 = st.columns((2.5, 3.5, 2.5), gap='medium')

    with col[0]:
        st.markdown('#### Total Mahasiswa')

        last_state_name = "Total Mahasiswa Terdaftar"  
        st.metric(label=last_state_name, value="2321", delta="231")
        
        first_state_name = "Total Mahasiswa Aktif"
        st.metric(label=first_state_name, value="2321", delta="231")

        last_state_name = "Total Mahasiswa Mangkir"  
        st.metric(label=last_state_name, value="2321", delta="232")

    with col[1]:
        st.markdown('#### Hubungan antara IPK dan Tahun Kelulusan')

        # Create scatter plot with color gradient
        fig = px.scatter(
            data_mahasiswa,
            x='IPK',
            y='Total Tahun',
            color='Total Tahun',
            color_continuous_scale='plasma',
            labels={'IPK': 'IPK', 'Total Tahun': 'Total Tahun'},
            opacity=0.7
        )

        # Calculate and add trend line using linear regression
        slope, intercept, _, _, _ = linregress(data_mahasiswa['IPK'], data_mahasiswa['Total Tahun'])
        regression_line = slope * data_mahasiswa['IPK'] + intercept
        fig.add_trace(
            go.Scatter(
                x=data_mahasiswa['IPK'],
                y=regression_line,
                mode='lines',
                name='Trend Line',
                line=dict(color='blue', width=2)
            )
        )

        # Update layout for clarity and style
        fig.update_layout(
            coloraxis_colorbar=dict(title='Total Tahun'),
            xaxis_title='IPK',
            yaxis_title='Total Tahun',
            template='plotly_white'
        )

        # Display plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)
    
    with col[2]:
        st.markdown('#### Presentase Kelulusan')

        # Example data for demonstration
        data = pd.DataFrame({
            'Category': ['Tepat Waktu', 'Over Study'],
            'Values': [317,123]
        })

        # Create pie chart
        fig = px.pie(
            data,
            names='Category',           # Column for category labels
            values='Values',             # Column for values
            hole=0.3                     # Optional: to create a donut chart effect
        )

        # Update the pie chart to display percentages
        fig.update_traces(textposition='inside', textinfo='percent+label')

        # Display the pie chart in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    with col2[0]:
        # 2. Scatter plot antara IPK dan Total Hari
        # List of columns to plot distribution for
        columns = ['Total Tahun']
        st.title("Distribusi Lama Studi Mahasiswa")

        # Loop through each column to create a separate interactive histogram
        for col in columns:
            fig = px.histogram(
                data_mahasiswa, 
                x=col, 
                nbins=20, 
                title=f'Distribusi {col}',
                labels={col: col, 'count': 'Frekuensi'},
                template='plotly_white'
            )
            
        # Customize hover information
        fig.update_traces(
            marker=dict(color='skyblue', line=dict(color='black', width=1)),
            hovertemplate=f'<b>{col}</b>: %{{x}}<br>Frekuensi: %{{y}}<extra></extra>'
        )
        # Display plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)
    with col2[1]:
        
        # 2. Scatter plot antara IPK dan Total Hari
        # List of columns to plot distribution for
        columns = ['IPK']
        st.title("Distribusi Frekuensi IPK Mahasiswa")

        # Loop through each column to create a separate interactive histogram
        for col in columns:
            fig = px.histogram(
                data_mahasiswa, 
                x=col, 
                nbins=20, 
                title=f'Distribusi {col}',
                labels={col: col, 'count': 'Frekuensi'},
                template='plotly_white'
            )
            
        # Customize hover information
        fig.update_traces(
            marker=dict(color='skyblue', line=dict(color='black', width=1)),
            hovertemplate=f'<b>{col}</b>: %{{x}}<br>Frekuensi: %{{y}}<extra></extra>'
        )
        
        # Display plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    with col2[2]:
        # 2. Scatter plot antara IPK dan Total Hari
        # List of columns to plot distribution for
        columns = ['Gol UKT']
        st.title("Distribusi Golongan UKT")

        # Loop through each column to create a separate interactive histogram
        for col in columns:
            fig = px.histogram(
                data_mahasiswa, 
                x=col, 
                nbins=20, 
                title=f'Distribusi {col}',
                labels={col: col, 'count': 'Frekuensi'},
                template='plotly_white'
            )
            
        # Customize hover information
        fig.update_traces(
            marker=dict(color='skyblue', line=dict(color='black', width=1)),
            hovertemplate=f'<b>{col}</b>: %{{x}}<br>Frekuensi: %{{y}}<extra></extra>'
        )
        
        # Display plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    # Define a simple color palette for Golongan UKT 0 to 7
    color_palette = px.colors.qualitative.Plotly  # A predefined color palette


    # Melting the DataFrame to long format for easier plotting
    ips_columns = ['IPS1', 'IPS2', 'IPS3', 'IPS4', 'IPS5', 'IPS6', 'IPS7', 'IPS8']
    melted_ips = data_mahasiswa.melt(id_vars=['Gol UKT'], value_vars=ips_columns, 
                                        var_name='Semester', value_name='IPS Score')

    avg_ips = melted_ips.groupby(['Gol UKT', 'Semester'])['IPS Score'].mean().reset_index()

    # Group by Gol UKT and Semester, and calculate the average score
    # avg_scores_by_ukt = df.groupby(['Gol UKT', 'Semester'])['Score'].mean().reset_index()

    # Create a line chart using Plotly
    fig = px.line(avg_ips, x='Semester', y='IPS Score', color='Gol UKT',
                title='Average IPS Scores by UKT Group',
                markers=True)

    # Customize layout
    fig.update_layout(
        xaxis_title='Semester',
        yaxis_title='Average IPS Score',
        title_font=dict(size=20, color='black'),
        font=dict(size=12, color='black'),
        hovermode='x unified'
    )

    # Streamlit app layout
    st.title('IPS Scores Based on Gol UKT')
    st.plotly_chart(fig)

elif selected == 'Clustering': 
    col = st.columns((4.25, 4.25), gap='medium')

    with col[0]:
        st.title("Cluster KMeans Visualization")

        # Define the features for clustering
        features = ['Gol UKT','IPK', 'Total Tahun']
        data_features = data_mahasiswa[features]

        # Standardize the features
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_features)

        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=2, random_state=42)
        clusters = kmeans.fit_predict(data_scaled)
        data_mahasiswa['Cluster'] = clusters

        # 3D Visualization using Plotly
        fig = px.scatter_3d(
            data_mahasiswa, x='Gol UKT', y='Total Tahun', z='IPK',
            color=data_mahasiswa['Cluster'].astype(str),  # Color by cluster
            color_discrete_sequence=px.colors.qualitative.Plotly,
            labels={'color': 'Cluster'},
            opacity=0.7,
            
        )
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(scene=dict(
            xaxis_title='Gol UKT',
            yaxis_title='Total Tahun',
            zaxis_title='IPK'
        ),
            width=900,  # Increase figure width
            height=700   # Increase figure height
        )
        st.plotly_chart(fig)

    with col[1]:
        # Display cluster assignments
        st.title("Cluster Statistics")
        # st.title("");
            # Detailed Descriptive Statistics for Each Cluster
        st.write("Descriptive Statistics for Each Cluster")
        cluster_descriptions = data_mahasiswa.groupby('Cluster').agg({
            'Gol UKT': ['mean', 'median', 'min', 'max'],
            'IPK': ['mean', 'median', 'min', 'max'],
            'Total Tahun': ['mean', 'median', 'min', 'max'],
            'IPS1': 'mean',
            'IPS2': 'mean',
            'IPS3': 'mean',
            'IPS4': 'mean',
            'IPS5': 'mean',
            'IPS6': 'mean',
            'IPS7': 'mean',
            'IPS8': 'mean'
        })
        st.dataframe(cluster_descriptions)
        st.write("Cluter Details")
        st.dataframe(data_mahasiswa[['NIM', 'Nama', 'Gol UKT', 'IPS1','IPS2','IPS3','IPS4','IPS5','IPK', 'Total Tahun', 'Cluster']])


