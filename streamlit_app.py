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
    
    year_list = list(df_reshaped.year.unique())[::-1]
    
    selected_year = st.selectbox('Select a year', year_list)
    df_selected_year = df_reshaped[df_reshaped.year == selected_year]

    color_theme_list = ['blues', 'cividis', 'greens', 'inferno', 'magma', 'plasma', 'reds', 'rainbow', 'turbo', 'viridis']
    selected_color_theme = st.selectbox('Select a color theme', color_theme_list)


#######################
# Choropleth map
def make_choropleth(input_df, input_id, input_column, input_color_theme):
    choropleth = px.choropleth(input_df, locations=input_id, color=input_column, locationmode="USA-states",
                               color_continuous_scale=input_color_theme,
                               range_color=(0, max(df_selected_year.population)),
                               scope="usa",
                               labels={'population':'Population'}
                              )
    choropleth.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        margin=dict(l=0, r=0, t=0, b=0),
        height=350
    )
    return choropleth


# Donut chart
def make_donut(input_response, input_text, input_color):
  if input_color == 'blue':
      chart_color = ['#29b5e8', '#155F7A']
  if input_color == 'green':
      chart_color = ['#27AE60', '#12783D']
  if input_color == 'orange':
      chart_color = ['#F39C12', '#875A12']
  if input_color == 'red':
      chart_color = ['#E74C3C', '#781F16']
    
  source = pd.DataFrame({
      "Topic": ['', input_text],
      "% value": [100-input_response, input_response]
  })
  source_bg = pd.DataFrame({
      "Topic": ['', input_text],
      "% value": [100, 0]
  })
    
  plot = alt.Chart(source).mark_arc(innerRadius=45, cornerRadius=25).encode(
      theta="% value",
      color= alt.Color("Topic:N",
                      scale=alt.Scale(
                          #domain=['A', 'B'],
                          domain=[input_text, ''],
                          # range=['#29b5e8', '#155F7A']),  # 31333F
                          range=chart_color),
                      legend=None),
  ).properties(width=130, height=130)
    
  text = plot.mark_text(align='center', color="#29b5e8", font="Lato", fontSize=32, fontWeight=700, fontStyle="italic").encode(text=alt.value(f'{input_response} %'))
  plot_bg = alt.Chart(source_bg).mark_arc(innerRadius=45, cornerRadius=20).encode(
      theta="% value",
      color= alt.Color("Topic:N",
                      scale=alt.Scale(
                          # domain=['A', 'B'],
                          domain=[input_text, ''],
                          range=chart_color),  # 31333F
                      legend=None),
  ).properties(width=130, height=130)
  return plot_bg + plot + text

# Convert population to text 
def format_number(num):
    if num > 1000000:
        if not num % 1000000:
            return f'{num // 1000000} M'
        return f'{round(num / 1000000, 1)} M'
    return f'{num // 1000} K'

# Calculation year-over-year population migrations
def calculate_population_difference(input_df, input_year):
  selected_year_data = input_df[input_df['year'] == input_year].reset_index()
  previous_year_data = input_df[input_df['year'] == input_year - 1].reset_index()
  selected_year_data['population_difference'] = selected_year_data.population.sub(previous_year_data.population, fill_value=0)
  return pd.concat([selected_year_data.states, selected_year_data.id, selected_year_data.population, selected_year_data.population_difference], axis=1).sort_values(by="population_difference", ascending=False)


#######################
# Dashboard Main Panel
col = st.columns((2.5, 3.5, 2.5), gap='medium')
col2 = st.columns((2.5, 3.5, 2.5), gap='medium')

with col[0]:
    st.markdown('#### Lulus Tepat Waktu')

    df_population_difference_sorted = calculate_population_difference(df_reshaped, selected_year)

    if selected_year > 2010:
        last_state_name = "Total Mahasiswa Terdaftar"
        last_state_population = format_number(df_population_difference_sorted.population.iloc[-1])   
        last_state_delta = format_number(df_population_difference_sorted.population_difference.iloc[-1])   
    else:
        last_state_name = '-'
        last_state_population = '-'
        last_state_delta = ''
    st.metric(label=last_state_name, value=last_state_population, delta=last_state_delta)
    
    if selected_year > 2010:
        first_state_name = "Total Mahasiswa Aktif"
        first_state_population = format_number(df_population_difference_sorted.population.iloc[0])
        first_state_delta = format_number(df_population_difference_sorted.population_difference.iloc[0])
    else:
        first_state_name = '-'
        first_state_population = '-'
        first_state_delta = ''
    st.metric(label=first_state_name, value=first_state_population, delta=first_state_delta)

    if selected_year > 2010:
        last_state_name = "Total Mahasiswa Mangkir"
        last_state_population = format_number(df_population_difference_sorted.population.iloc[-1])   
        last_state_delta = format_number(df_population_difference_sorted.population_difference.iloc[-1])   
    else:
        last_state_name = '-'
        last_state_population = '-'
        last_state_delta = ''
    st.metric(label=last_state_name, value=last_state_population, delta=last_state_delta)

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


st.sidebar.subheader("2. Example Dataset")
dataset_name = st.sidebar.selectbox(
    "Default Data", [dn.value for dn in DATASET_NAMES])
is_3d = st.sidebar.checkbox("Use 3D Features", value=False)
default_dataset_points = get_default_dataset_points(dataset_name, is_3d)

cluster_features = add_user_data_input_listener(default_dataset_points)
# normalize dataset for easier parameter selection
cluster_features_scaled = StandardScaler().fit_transform(cluster_features)

st.title("Cluster Visualization")
st.write("Streamlit application of the [sklearn cluster comparison](https://scikit-learn.org/stable/modules/clustering.html) page. \n"
            "Feel free to upload your own custom data or to play around with one of the example datasets. The parameters of the cluster algorithm can be updated interactively and three-dimensional visualizations are also possible.")

# Cluster Algo
cluster_algos: List[str] = st.sidebar.multiselect(
    "Cluster Algorithms", [ca.value for ca in CLUSTER_ALGORITHMS], [CLUSTER_ALGORITHMS[0]])
for cluster_algo_splitted_list in split_list(cluster_algos, MAX_PLOTS_PER_ROW):
    display_cols = st.columns(1)
    for i, cluster_algo_str in enumerate(cluster_algo_splitted_list):
        st.sidebar.title(cluster_algo_str)
        cluster_algo_kwargs = get_cluster_algo_parameters(cluster_algo_str, cluster_features_scaled, dataset_name)
        cluster_labels = get_cluster_labels(cluster_features_scaled, cluster_algo_str, **cluster_algo_kwargs)

        # Visualize clustering results
        display_cols[0].subheader(cluster_algo_str)
        fig = plot_figure(cluster_features,cluster_labels)

        # prevent that a single visualization does not take the whole width
        use_container_width = len(cluster_algos) > 1
        display_cols[0].plotly_chart(fig, use_container_width=use_container_width)




