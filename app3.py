import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set page config
st.set_page_config(page_title="Developer Dashboard", layout="wide")


st.sidebar.title("Developers Coaching Dashboard ")
page = st.sidebar.radio("Go to", ["Pull Request Analysis", "Efficiency Analysis", "Predictive Model Analysis"])



def predicitive_model():
    """
        This function builds a predictive model to analyze and cluster developers' performance.
        It applies K-Means clustering on key performance metrics like merge time and review time.
        The elbow method helps determine the optimal number of clusters for better segmentation.
        A scatter plot visualizes clusters based on merge and review times, making insights clear.
        A bar chart displays the distribution of developers across clusters for quick interpretation.
        K-Nearest Neighbors (KNN) is used to classify developers into clusters and evaluate accuracy.
        Finally, key performance metrics such as accuracy, precision, recall, and F1-score are shown.
    """

    @st.cache_data
    def load_data():
        df = pd.read_csv("https://raw.githubusercontent.com/Addika1630/semantic-search/refs/heads/main/static/featured_data-update.csv")  
        df["updated_at"] = pd.to_datetime(df["updated_at"])  
        return df

    df = load_data()

    features = ['merge_time_hours', 'review_time_hours', 'pickup_time_hours', 'review_cycles_count']

    df_grouped = df.groupby("owner")[features].mean().reset_index()

    df_grouped.fillna(df_grouped.median(numeric_only=True), inplace=True)

    st.title("🔹 Forecast Developers' Performance")

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_grouped[features])

    num_clusters = st.sidebar.slider("Select Number of Clusters", min_value=2, max_value=10, value=6)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df_grouped['Cluster'] = kmeans.fit_predict(df_scaled)
    df_grouped['Cluster'] = df_grouped['Cluster'].astype(str)  

    st.sidebar.subheader("Filter by Cluster")
    selected_cluster = st.sidebar.selectbox("Select Cluster", ["All"] + sorted(df_grouped["Cluster"].unique().tolist()))

    if selected_cluster != "All":
        df_filtered = df_grouped[df_grouped["Cluster"] == selected_cluster]
    else:
        df_filtered = df_grouped

    
    fig_scatter = px.scatter(df_filtered, x="merge_time_hours", y="review_time_hours", color="Cluster",
                            hover_data=['owner', 'pickup_time_hours', 'review_cycles_count'],
                            title="Clustered Visualization of Developers' Performance",
                            text="owner")  

    fig_scatter.update_layout(
        title={
            'font': {'size': 24},  
            'x': 0.5,  
            'xanchor': 'center'
        },
        template="plotly_dark",  
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="right", x=1)
    )

    fig_scatter.update_traces(marker=dict(size=8, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')),
                              textposition="top center")
    st.plotly_chart(fig_scatter, use_container_width=True)

    cluster_counts = df_grouped['Cluster'].value_counts().reset_index()
    cluster_counts.columns = ['Cluster', 'Count']
    fig_bar = px.bar(cluster_counts, x='Cluster', y='Count', title="Count of Developers' on each Cluster", color='Cluster')
    fig_bar.update_layout(
        title={
            'font': {'size': 24},  
            'x': 0.5,  
            'xanchor': 'center'
        },
        template="plotly_dark"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    @st.cache_data
    def calculate_wcss(data, max_clusters=10):
        wcss = []
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(data)
            wcss.append(kmeans.inertia_)  
        return wcss

    wcss_values = calculate_wcss(df_scaled, max_clusters=10)

    fig_elbow = px.line(x=range(1, 11), y=wcss_values, markers=True,
                        title="Elbow Method for Finding Optimal Number of Clusters",
                        labels={"x": "Number of Clusters (k)", "y": "WCSS"})
    fig_elbow.add_scatter(x=[2, 3, 4, 5], y=wcss_values[1:5], mode="markers", marker=dict(size=10, color="red"))
    fig_elbow.update_layout(
        title={
            'font': {'size': 24}, 
            'x': 0.5,  
            'xanchor': 'center'
        },
        template="plotly_dark"
    )
    st.plotly_chart(fig_elbow, use_container_width=True)

    X = df_grouped[features]
    y = df_grouped['Cluster']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    st.subheader("KNN Performance Metrics Result")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)

    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"F1 Score: {f1:.2f}")

    if st.checkbox("Show Raw Data"):
        st.dataframe(df_filtered)


def efficiency_analysis():
    """
    Analyzes pull request efficiency over time.
    Enhancements:
    - Interactive hover effects for better readability
    - Dark & light mode compatibility
    - Gridlines & background styling for clarity
    - Optimized legend positioning
    """

    analysis_df = pd.read_csv("https://raw.githubusercontent.com/Addika1630/semantic-search/refs/heads/main/static/featured_data-update.csv")
    analysis_df['updated_at'] = pd.to_datetime(analysis_df['updated_at'])
    analysis_df.set_index('updated_at', inplace=True)

    weekly_avg_df = analysis_df.resample('25D').agg({
        'total_time_hours': 'mean',
        'merge_time_hours': 'mean',
        'pickup_time_hours': 'mean',
        'review_time_hours': 'mean',
        'review_cycles_count': 'mean'  
    })
    weekly_avg_df.reset_index(inplace=True)

    st.title('📊 Pull Request Efficiency Analysis')
    st.markdown("### Trends in Merge, Pickup, and Review Times")

    df_melted = weekly_avg_df.melt(id_vars=['updated_at'],
                                   value_vars=['total_time_hours', 'merge_time_hours', 'pickup_time_hours', 'review_time_hours'],
                                   var_name='Time Variable',
                                   value_name='Time (hours)')
    
    color_map = {
        'total_time_hours': 'blue',
        'merge_time_hours': 'green',
        'pickup_time_hours': 'orange',
        'review_time_hours': 'red'
    }
    
    fig = go.Figure()
    for var in color_map.keys():
        fig.add_trace(go.Scatter(
            x=df_melted[df_melted['Time Variable'] == var]['updated_at'],
            y=df_melted[df_melted['Time Variable'] == var]['Time (hours)'],
            mode='lines+markers',
            name=var.replace('_', ' ').title(),
            line=dict(color=color_map[var], shape='spline', smoothing=1.3),
            marker=dict(size=6, symbol='circle')
        ))

    fig.update_layout(
        title='⏳ Averaged Time Variables vs Date',
        xaxis_title='Date',
        yaxis_title='Time (hours)',
        template='plotly_white',  
        legend=dict(title='Time Variables', orientation='h', x=0.5, xanchor='center'),
        hovermode='x unified',
        width=1200, height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Review Cycles Trend Over Time")
    fig_review_cycles = go.Figure()
    fig_review_cycles.add_trace(go.Scatter(
        x=weekly_avg_df['updated_at'],
        y=weekly_avg_df['review_cycles_count'],
        mode='lines+markers',
        name='Review Cycles Count',
        line=dict(color='purple', shape='spline', smoothing=1.3),
        marker=dict(size=6, symbol='diamond')
    ))

    fig_review_cycles.update_layout(
        title='🔄 Review Cycles Count vs Date',
        xaxis_title='Date',
        yaxis_title='Review Cycles Count',
        template='plotly_white',
        legend=dict(x=0.02, y=0.98, bordercolor='black', borderwidth=1),
        hovermode='x unified',
        width=1200, height=600
    )
    
    st.plotly_chart(fig_review_cycles, use_container_width=True)


def pull_request_analysis():
    """
        This function, `pull_request_analysis()`, is a Streamlit-based web app for analyzing 
        developer pull requests (PRs) using data. It provides interactive filtering by developer,
        visualizations of PR trends over time, and a display of raw data. The app features a 
        responsive UI with styled buttons and uses session state to retain user selections. 
        A moving average smoothing technique is applied to visualize PR trends more clearly. 
        Users can toggle between different views, such as filtered PRs and overall trends.
    """

    st.markdown("""
    <style>
        /* Button styles */
        .stButton>button {
            background-color: #444444;
            color: white;
            border: none;
            padding: 6px 12px;  /* Reduced padding for smaller button */
            border-radius: 12px;  /* Rounded corners */
            font-size: 12px;  /* Reduced font size */
            font-weight: bold;
            text-align: center;
            transition: all 0.3s ease;  /* Smooth transition for hover effects */
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);  /* Subtle shadow for depth */
            cursor: pointer;
            width: 200px; 
        }

        .stButton>button:hover {
            background-color: #666666;
            transform: scale(1.05);  /* Slight scaling effect on hover */
            box-shadow: 0px 6px 12px rgba(0, 0, 0, 0.3);  /* Increased shadow on hover */
        }

        .stButton>button:focus {
            outline: none;
        }

        /* Adjust color and shadow for button when clicked */
        .stButton>button:active {
            background-color: #555555;
            transform: scale(1.1);
        }
    </style>
    """, unsafe_allow_html=True)

    def load_data():
        df = pd.read_csv("https://raw.githubusercontent.com/Addika1630/semantic-search/refs/heads/main/static/cleaned_data.csv")
        df["updated_at"] = pd.to_datetime(df["updated_at"], unit='s')
        return df

    if "selected_owner" not in st.session_state:
        st.session_state.selected_owner = None

    def select_owner(owner):
        st.session_state.selected_owner = owner

    def reset_owner_selection():
        st.session_state.selected_owner = None

    def main():
        st.title("📊 Developers' Pull Request Analysis")

        df = load_data()
        merged_prs = df[df["action"] == "PR_MERGED"]

        total_merged = merged_prs.shape[0]
        st.metric("Total Merged PRs", total_merged)

        if st.button("Show All"):
            reset_owner_selection()

        st.subheader("Select an Owner to Filter Data")
        top_owners = merged_prs["owner"].value_counts().head(20)
        
        cols = st.columns(5)
        for i, (owner, count) in enumerate(top_owners.items()):
            with cols[i % 5]:
                if st.button(f"{owner} ({count})", key=owner, use_container_width=False):
                    select_owner(owner)

        if st.session_state.selected_owner:
            st.subheader(f"Showing Data for: {st.session_state.selected_owner}")
            filtered_prs = merged_prs[merged_prs["owner"] == st.session_state.selected_owner]
        else:
            filtered_prs = merged_prs

        merged_over_time = filtered_prs.resample("4D", on="updated_at").size()

        window_size = 4  
        smoothed_merged_over_time = merged_over_time.rolling(window=window_size).mean()

        fig = px.line(x=smoothed_merged_over_time.index, 
                    y=smoothed_merged_over_time.values, 
                    labels={'x': 'Date', 'y': 'Number of PRs Merged'},
                    title="Merged PRs Over Time")

        fig.update_traces(line_shape='spline')

        fig.update_layout(
            height=500,  
            title_x=0.5,  
            title_y=0.95  
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # Show raw data
        if st.checkbox("Show Raw Data"):
            st.dataframe(filtered_prs)
        
    main()

# Page Controller
if page == "Pull Request Analysis":
    pull_request_analysis()
elif page == "Efficiency Analysis":
    efficiency_analysis()
elif page == "Predictive Model Analysis":
    predicitive_model()


