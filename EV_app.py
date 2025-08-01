import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set Streamlit page config
st.set_page_config(
    page_title="EV Adoption Forecaster", 
    page_icon="🔮", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
        .main-header {
            text-align: center;
            font-size: 42px;
            font-weight: bold;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        .sub-header {
            text-align: center;
            font-size: 20px;
            color: #666;
            margin-bottom: 30px;
        }
        .metric-container {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin: 10px 0;
        }
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        }
        .stSelectbox > div > div > select {
            background-color: #f0f2f6;
        }
    </style>
""", unsafe_allow_html=True)

# File paths - Update these to your local directory
BASE_PATH = r"C:\Users\bnaik\Downloads\EV project"
MODEL_PATH = f"{BASE_PATH}\\forecasting_ev_model.pkl"
ENCODER_PATH = f"{BASE_PATH}\\label_encoder.pkl"
DATA_PATH = f"{BASE_PATH}\\Electric_Vehicle_Population_Size_History_By_County_.csv"

# Function to preprocess data (based on code 2)
@st.cache_data
def preprocess_data(df):
    """Enhanced data preprocessing following code 2 methodology"""
    
    # Outlier handling for Percent Electric Vehicles
    Q1 = df['Percent Electric Vehicles'].quantile(0.25)
    Q3 = df['Percent Electric Vehicles'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Cap outliers (winsorization)
    df['Percent Electric Vehicles'] = np.where(
        df['Percent Electric Vehicles'] > upper_bound, upper_bound,
        np.where(df['Percent Electric Vehicles'] < lower_bound, lower_bound, 
                 df['Percent Electric Vehicles'])
    )
    
    # Date processing
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df[df['Date'].notnull()]
    
    # Remove rows where target is missing
    df = df[df['Electric Vehicle (EV) Total'].notnull()]
    
    # Fill missing categorical values
    df['County'] = df['County'].fillna('Unknown')
    df['State'] = df['State'].fillna('Unknown')
    
    # Convert numeric columns
    cols_to_convert = [
        'Battery Electric Vehicles (BEVs)',
        'Plug-In Hybrid Electric Vehicles (PHEVs)',
        'Electric Vehicle (EV) Total',
        'Non-Electric Vehicle Total',
        'Total Vehicles',
        'Percent Electric Vehicles'
    ]
    
    for col in cols_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Time-based features
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['numeric_date'] = df['Date'].dt.year * 12 + df['Date'].dt.month
    
    # Encode categorical features
    le = LabelEncoder()
    df['county_encoded'] = le.fit_transform(df['County'])
    
    # Sort data for time series feature creation
    df = df.sort_values(['County', 'Date'])
    df['months_since_start'] = df.groupby('County').cumcount()
    
    # Create lag features (1-3 months)
    for lag in [1, 2, 3]:
        df[f'ev_total_lag{lag}'] = df.groupby('County')['Electric Vehicle (EV) Total'].shift(lag)
    
    # Rolling average (3-month)
    df['ev_total_roll_mean_3'] = df.groupby('County')['Electric Vehicle (EV) Total'] \
                                   .transform(lambda x: x.shift(1).rolling(3).mean())
    
    # Percent change features
    df['ev_total_pct_change_1'] = df.groupby('County')['Electric Vehicle (EV) Total'] \
                                    .pct_change(periods=1, fill_method=None)
    df['ev_total_pct_change_3'] = df.groupby('County')['Electric Vehicle (EV) Total'] \
                                    .pct_change(periods=3, fill_method=None)
    
    # Clean infinite values
    df['ev_total_pct_change_1'] = df['ev_total_pct_change_1'].replace([np.inf, -np.inf], np.nan).fillna(0)
    df['ev_total_pct_change_3'] = df['ev_total_pct_change_3'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Cumulative and growth slope features
    df['cumulative_ev'] = df.groupby('County')['Electric Vehicle (EV) Total'].cumsum()
    
    df['ev_growth_slope'] = df.groupby('County')['cumulative_ev'].transform(
        lambda x: x.rolling(6).apply(lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) == 6 else np.nan)
    )
    
    # Remove rows with missing lag features
    df = df.dropna().reset_index(drop=True)
    
    return df, le

# Function to train model if not exists
@st.cache_resource
def train_or_load_model():
    """Train model or load existing one"""
    try:
        # Try to load existing model
        model = joblib.load(MODEL_PATH)
        label_encoder = joblib.load(ENCODER_PATH)
        st.success("✅ Loaded existing trained model!")
        return model, label_encoder, None
    except FileNotFoundError:
        st.warning("⚠️ No existing model found. Training new model...")
        
        # Load and preprocess data
        try:
            df = pd.read_csv(DATA_PATH)
            df, le = preprocess_data(df)
            
            # Define features and target (matching code 2)
            features = [
                'months_since_start',
                'county_encoded',
                'ev_total_lag1',
                'ev_total_lag2',
                'ev_total_lag3',
                'ev_total_roll_mean_3',
                'ev_total_pct_change_1',
                'ev_total_pct_change_3',
                'ev_growth_slope',
            ]
            
            target = 'Electric Vehicle (EV) Total'
            X = df[features]
            y = df[target]
            
            # Train-test split (time-aware)
            X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.1)
            
            # Train Random Forest model
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=42
            )
            
            with st.spinner("Training model..."):
                model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Save model
            joblib.dump(model, MODEL_PATH)
            joblib.dump(le, ENCODER_PATH)
            
            st.success(f"✅ Model trained successfully! MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
            
            return model, le, df
            
        except Exception as e:
            st.error(f"❌ Error training model: {e}")
            st.stop()

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess data"""
    try:
        df = pd.read_csv(DATA_PATH)
        df, le = preprocess_data(df)
        return df
    except FileNotFoundError:
        st.error(f"❌ Data file not found: {DATA_PATH}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.stop()

# Load model and data
model, label_encoder, _ = train_or_load_model()
df = load_and_preprocess_data()

# Define features (matching code 2)
features = [
    'months_since_start',
    'county_encoded',
    'ev_total_lag1',
    'ev_total_lag2',
    'ev_total_lag3',
    'ev_total_roll_mean_3',
    'ev_total_pct_change_1',
    'ev_total_pct_change_3',
    'ev_growth_slope',
]

# Header
st.markdown('<h1 class="main-header">🔮 EV Adoption Forecasting Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Electric Vehicle adoption prediction using Random Forest ML model</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("🎯 Forecast Configuration")
st.sidebar.markdown("---")

# Get available counties
county_list = sorted(df['County'].dropna().unique().tolist())
total_counties = len(county_list)

# County selection
selected_county = st.sidebar.selectbox(
    "🏘️ Select County for Detailed Forecast",
    county_list,
    index=county_list.index("King") if "King" in county_list else 0
)

# Forecast parameters
forecast_months = st.sidebar.slider(
    "📅 Forecast Period (months)",
    min_value=6,
    max_value=60,
    value=36,
    step=6
)

# Show dataset info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Information")
st.sidebar.info(f"""
**Total Counties:** {total_counties}  
**Date Range:** {df['Date'].min().strftime('%Y-%m')} to {df['Date'].max().strftime('%Y-%m')}  
**Total Records:** {len(df):,}  
**Features Used:** {len(features)} engineered features
""")

# Main content area
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Single County Forecast", "🔄 Multi-County Comparison", "📊 Analytics Dashboard", "🔬 Data Analysis", "ℹ️ Model Info"])

# Tab 1: Single County Forecast (Enhanced with code 2 methodology)
with tab1:
    st.header(f"🎯 Detailed Forecast for {selected_county} County")
    
    # Validate county exists
    if selected_county not in df['County'].unique():
        st.warning(f"❌ County '{selected_county}' not found in dataset.")
        st.stop()
    
    # Filter data for selected county
    county_df = df[df['County'] == selected_county].sort_values("Date")
    
    if len(county_df) < 6:
        st.warning(f"❌ Insufficient data for {selected_county} County (need at least 6 months)")
        st.stop()
    
    # Current statistics
    col1, col2, col3, col4 = st.columns(4)
    
    current_total = county_df['Electric Vehicle (EV) Total'].iloc[-1]
    total_growth = county_df['Electric Vehicle (EV) Total'].iloc[-1] - county_df['Electric Vehicle (EV) Total'].iloc[0]
    avg_monthly = county_df['Electric Vehicle (EV) Total'].mean()
    latest_date = county_df['Date'].max()
    
    with col1:
        st.metric("Current EV Total", f"{current_total:,.0f}", f"+{total_growth:,.0f}")
    with col2:
        st.metric("Average Monthly", f"{avg_monthly:,.0f}")
    with col3:
        st.metric("Latest Data", latest_date.strftime('%Y-%m'))
    with col4:
        st.metric("Forecast Period", f"{forecast_months} months")
    
    # Enhanced forecast generation (following code 2 methodology)
    county_code = county_df['county_encoded'].iloc[0]
    historical_ev = list(county_df['Electric Vehicle (EV) Total'].values[-6:])
    cumulative_ev = list(np.cumsum(historical_ev))
    months_since_start = county_df['months_since_start'].max()
    latest_date = county_df['Date'].max()
    
    future_rows = []
    
    # Progress bar for forecast generation
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(1, forecast_months + 1):
        progress_bar.progress(i / forecast_months)
        status_text.text(f'Generating forecast... {i}/{forecast_months} months')
        
        forecast_date = latest_date + pd.DateOffset(months=i)
        months_since_start += 1
        
        # Calculate features (matching code 2)
        lag1, lag2, lag3 = historical_ev[-1], historical_ev[-2], historical_ev[-3]
        roll_mean = np.mean([lag1, lag2, lag3])
        pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
        pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
        
        # Calculate growth slope
        recent_cumulative = cumulative_ev[-6:]
        ev_growth_slope = np.polyfit(range(len(recent_cumulative)), recent_cumulative, 1)[0] if len(recent_cumulative) == 6 else 0
        
        # Feature vector
        feature_dict = {
            'months_since_start': months_since_start,
            'county_encoded': county_code,
            'ev_total_lag1': lag1,
            'ev_total_lag2': lag2,
            'ev_total_lag3': lag3,
            'ev_total_roll_mean_3': roll_mean,
            'ev_total_pct_change_1': pct_change_1,
            'ev_total_pct_change_3': pct_change_3,
            'ev_growth_slope': ev_growth_slope
        }
        
        # Predict
        X_new = pd.DataFrame([feature_dict])[features]
        pred = model.predict(X_new)[0]
        
        future_rows.append({
            "Date": forecast_date, 
            "Predicted_EV_Total": max(0, round(pred)),  # Ensure non-negative
            "Month": forecast_date.strftime('%Y-%m')
        })
        
        # Update rolling history
        historical_ev.append(pred)
        if len(historical_ev) > 6:
            historical_ev.pop(0)
        
        cumulative_ev.append(cumulative_ev[-1] + pred)
        if len(cumulative_ev) > 6:
            cumulative_ev.pop(0)
    
    progress_bar.empty()
    status_text.empty()
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame(future_rows)
    
    # Combine historical and forecast for plotting
    historical_data = county_df[['Date', 'Electric Vehicle (EV) Total']].copy()
    historical_data['Source'] = 'Historical'
    historical_data['Cumulative_EV'] = historical_data['Electric Vehicle (EV) Total'].cumsum()
    
    forecast_plot_data = forecast_df.copy()
    forecast_plot_data['Source'] = 'Forecast'
    forecast_plot_data['Cumulative_EV'] = forecast_plot_data['Predicted_EV_Total'].cumsum() + historical_data['Cumulative_EV'].iloc[-1]
    
    # Interactive Plot using Plotly
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Monthly EV Adoption', 'Cumulative EV Growth'),
        vertical_spacing=0.12
    )
    
    # Monthly plot
    fig.add_trace(
        go.Scatter(
            x=historical_data['Date'],
            y=historical_data['Electric Vehicle (EV) Total'],
            mode='lines+markers',
            name='Historical',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=6)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=forecast_plot_data['Date'],
            y=forecast_plot_data['Predicted_EV_Total'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#ff7f0e', width=3, dash='dash'),
            marker=dict(size=6, symbol='diamond')
        ),
        row=1, col=1
    )
    
    # Cumulative plot
    fig.add_trace(
        go.Scatter(
            x=historical_data['Date'],
            y=historical_data['Cumulative_EV'],
            mode='lines+markers',
            name='Historical (Cumulative)',
            line=dict(color='#2ca02c', width=3),
            marker=dict(size=6),
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=forecast_plot_data['Date'],
            y=forecast_plot_data['Cumulative_EV'],
            mode='lines+markers',
            name='Forecast (Cumulative)',
            line=dict(color='#d62728', width=3, dash='dash'),
            marker=dict(size=6, symbol='diamond'),
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f"EV Adoption Forecast - {selected_county} County",
        height=700,
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Monthly EV Count", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative EV Count", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Forecast Summary
    historical_total = historical_data['Cumulative_EV'].iloc[-1]
    forecasted_total = forecast_plot_data['Cumulative_EV'].iloc[-1]
    growth_absolute = forecasted_total - historical_total
    growth_percentage = (growth_absolute / historical_total) * 100 if historical_total > 0 else 0
    
    st.markdown("### 📋 Forecast Summary")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h3>📊 Current Status</h3>
            <h2>{historical_total:,.0f}</h2>
            <p>Total EVs (Current)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <h3>🎯 Projected Total</h3>
            <h2>{forecasted_total:,.0f}</h2>
            <p>Total EVs (Forecasted)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        trend_icon = "📈" if growth_percentage > 0 else "📉"
        st.markdown(f"""
        <div class="metric-container">
            <h3>{trend_icon} Growth Rate</h3>
            <h2>{growth_percentage:.1f}%</h2>
            <p>Over {forecast_months} months</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Downloadable forecast data
    st.markdown("### 💾 Download Forecast Data")
    csv_data = forecast_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Forecast as CSV",
        data=csv_data,
        file_name=f"{selected_county}_EV_Forecast_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# Tab 2: Multi-County Comparison (Enhanced)
with tab2:
    st.header("🔄 Multi-County Comparison")
    
    # County selection for comparison
    comparison_counties = st.multiselect(
        "Select counties to compare (up to 5):",
        county_list,
        default=["King", "Pierce", "Snohomish"] if all(c in county_list for c in ["King", "Pierce", "Snohomish"]) else county_list[:3],
        max_selections=5
    )
    
    if comparison_counties:
        comparison_data = []
        
        # Progress bar for multi-county processing
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, county in enumerate(comparison_counties):
            progress_bar.progress((idx + 1) / len(comparison_counties))
            status_text.text(f'Processing {county} County... ({idx + 1}/{len(comparison_counties)})')
            
            county_df = df[df['County'] == county].sort_values("Date")
            if len(county_df) < 6:
                continue
                
            county_code = county_df['county_encoded'].iloc[0]
            
            # Generate forecast for each county (following code 2 methodology)
            hist_ev = list(county_df['Electric Vehicle (EV) Total'].values[-6:])
            cum_ev = list(np.cumsum(hist_ev))
            months_since = county_df['months_since_start'].max()
            last_date = county_df['Date'].max()
            
            future_rows_county = []
            for i in range(1, forecast_months + 1):
                forecast_date = last_date + pd.DateOffset(months=i)
                months_since += 1
                
                lag1, lag2, lag3 = hist_ev[-1], hist_ev[-2], hist_ev[-3]
                roll_mean = np.mean([lag1, lag2, lag3])
                pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
                pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
                recent_cum = cum_ev[-6:]
                ev_slope = np.polyfit(range(len(recent_cum)), recent_cum, 1)[0] if len(recent_cum) == 6 else 0
                
                feature_dict = {
                    'months_since_start': months_since,
                    'county_encoded': county_code,
                    'ev_total_lag1': lag1,
                    'ev_total_lag2': lag2,
                    'ev_total_lag3': lag3,
                    'ev_total_roll_mean_3': roll_mean,
                    'ev_total_pct_change_1': pct_change_1,
                    'ev_total_pct_change_3': pct_change_3,
                    'ev_growth_slope': ev_slope
                }
                
                X_new = pd.DataFrame([feature_dict])[features]
                pred = model.predict(X_new)[0]
                future_rows_county.append({"Date": forecast_date, "Predicted_EV_Total": max(0, round(pred))})
                
                hist_ev.append(pred)
                if len(hist_ev) > 6:
                    hist_ev.pop(0)
                
                cum_ev.append(cum_ev[-1] + pred)
                if len(cum_ev) > 6:
                    cum_ev.pop(0)
            
            # Combine historical and forecast
            hist_cum = county_df[['Date', 'Electric Vehicle (EV) Total']].copy()
            hist_cum['Cumulative_EV'] = hist_cum['Electric Vehicle (EV) Total'].cumsum()
            
            fc_df = pd.DataFrame(future_rows_county)
            fc_df['Cumulative_EV'] = fc_df['Predicted_EV_Total'].cumsum() + hist_cum['Cumulative_EV'].iloc[-1]
            
            combined_county = pd.concat([
                hist_cum[['Date', 'Cumulative_EV']],
                fc_df[['Date', 'Cumulative_EV']]
            ], ignore_index=True)
            
            combined_county['County'] = county
            comparison_data.append(combined_county)
        
        progress_bar.empty()
        status_text.empty()
        
        if comparison_data:
            # Combine all counties data
            comp_df = pd.concat(comparison_data, ignore_index=True)
            
            # Interactive comparison plot
            fig = px.line(
                comp_df, 
                x='Date', 
                y='Cumulative_EV', 
                color='County',
                title=f"EV Adoption Comparison: {len(comparison_counties)} Counties",
                labels={'Cumulative_EV': 'Cumulative EV Count', 'Date': 'Date'}
            )
            
            fig.update_layout(
                height=600,
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Growth comparison table
            st.markdown("### 📊 Growth Comparison Summary")
            
            growth_summary = []
            for county in comparison_counties:
                county_data = comp_df[comp_df['County'] == county].reset_index(drop=True)
                if len(county_data) > forecast_months:
                    historical_total = county_data['Cumulative_EV'].iloc[-(forecast_months + 1)]
                    forecasted_total = county_data['Cumulative_EV'].iloc[-1]
                    growth_pct = ((forecasted_total - historical_total) / historical_total) * 100 if historical_total > 0 else 0
                    
                    growth_summary.append({
                        'County': county,
                        'Current Total': f"{historical_total:,.0f}",
                        'Projected Total': f"{forecasted_total:,.0f}",
                        'Growth (%)': f"{growth_pct:.1f}%",
                        'Growth Trend': "📈" if growth_pct > 0 else "📉"
                    })
            
            if growth_summary:
                summary_df = pd.DataFrame(growth_summary)
                st.dataframe(summary_df, use_container_width=True)

# Tab 3: Analytics Dashboard
with tab3:
    st.header("📊 Analytics Dashboard")
    
    # Overall statistics
    st.markdown("### 🎯 State-wide EV Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_evs = df['Electric Vehicle (EV) Total'].sum()
    total_bevs = df['Battery Electric Vehicles (BEVs)'].sum() if 'Battery Electric Vehicles (BEVs)' in df.columns else 0
    total_phevs = df['Plug-In Hybrid Electric Vehicles (PHEVs)'].sum() if 'Plug-In Hybrid Electric Vehicles (PHEVs)' in df.columns else 0
    avg_percent = df['Percent Electric Vehicles'].mean()
    
    with col1:
        st.metric("Total EVs", f"{total_evs:,.0f}")
    with col2:
        st.metric("Battery EVs", f"{total_bevs:,.0f}")
    with col3:
        st.metric("Plug-in Hybrids", f"{total_phevs:,.0f}")
    with col4:
        st.metric("Avg EV %", f"{avg_percent:.2f}%")
    
    # County rankings
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏆 Top 10 Counties by EV Count")
        top_counties = df.groupby('County')['Electric Vehicle (EV) Total'].sum().sort_values(ascending=False).head(10)
        
        fig = px.bar(
            x=top_counties.values,
            y=top_counties.index,
            orientation='h',
            title="Top Counties by Total EV Count",
            labels={'x': 'Total EV Count', 'y': 'County'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 EV Percentage Leaders")
        top_percent = df.groupby('County')['Percent Electric Vehicles'].mean().sort_values(ascending=False).head(10)
        
        fig = px.bar(
            x=top_percent.values,
            y=top_percent.index,
            orientation='h',
            title="Top Counties by EV Percentage",
            labels={'x': 'EV Percentage', 'y': 'County'},
            color=top_percent.values,
            color_continuous_scale="viridis"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Time series analysis
    st.markdown("### 📅 State-wide EV Growth Over Time")
    
    monthly_growth = df.groupby('Date')['Electric Vehicle (EV) Total'].sum().reset_index()
    monthly_growth['Cumulative'] = monthly_growth['Electric Vehicle (EV) Total'].cumsum()
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Monthly EV Additions', 'Cumulative EV Growth')
    )
    
    fig.add_trace(
        go.Scatter(
            x=monthly_growth['Date'],
            y=monthly_growth['Electric Vehicle (EV) Total'],
            mode='lines+markers',
            name='Monthly',
            line=dict(color='#1f77b4', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=monthly_growth['Date'],
            y=monthly_growth['Cumulative'],
            mode='lines+markers',
            name='Cumulative',
            line=dict(color='#ff7f0e', width=2)
        ),
        row=1, col=2
    )
    
    fig.update_layout(height=400, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

# Tab 4: Data Analysis (New tab based on code 2)
with tab4:
    st.header("🔬 Data Analysis & Insights")
    
    # Outlier analysis section
    st.markdown("### 📊 Data Quality Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Outlier Treatment Results")
        Q1 = df['Percent Electric Vehicles'].quantile(0.25)
        Q3 = df['Percent Electric Vehicles'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        st.info(f"""
        **Outlier Boundaries:**  
        Lower bound: {lower_bound:.4f}%  
        Upper bound: {upper_bound:.4f}%  
        IQR: {IQR:.4f}
        """)
    
    with col2:
        st.markdown("#### Dataset Statistics")
        st.info(f"""
        **Total Records:** {len(df):,}  
        **Counties:** {df['County'].nunique()}  
        **Date Range:** {df['Date'].min().strftime('%Y-%m')} to {df['Date'].max().strftime('%Y-%m')}  
        **Avg EV %:** {df['Percent Electric Vehicles'].mean():.2f}%
        """)
    
    # Vehicle distribution visualization (from code 2)
    st.markdown("### 🚗 Vehicle Fleet Distribution")
    
    # Calculate totals
    bev_total = df['Battery Electric Vehicles (BEVs)'].fillna(0).sum() if 'Battery Electric Vehicles (BEVs)' in df.columns else 0
    phev_total = df['Plug-In Hybrid Electric Vehicles (PHEVs)'].fillna(0).sum() if 'Plug-In Hybrid Electric Vehicles (PHEVs)' in df.columns else 0
    ev_total = df['Electric Vehicle (EV) Total'].fillna(0).sum()
    non_ev_total = df['Non-Electric Vehicle Total'].fillna(0).sum() if 'Non-Electric Vehicle Total' in df.columns else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("BEV Total", f"{bev_total:,.0f}")
    with col2:
        st.metric("PHEV Total", f"{phev_total:,.0f}")
    with col3:
        st.metric("EV Total", f"{ev_total:,.0f}")
    with col4:
        st.metric("Non-EV Total", f"{non_ev_total:,.0f}")
    
    # Vehicle type breakdown charts
    if bev_total > 0 or phev_total > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # EV Type Distribution
            ev_types = ['BEV', 'PHEV']
            ev_counts = [bev_total, phev_total]
            
            fig = px.bar(
                x=ev_types, 
                y=ev_counts,
                title="Electric Vehicle Type Distribution",
                labels={'x': 'Vehicle Type', 'y': 'Count'},
                color=ev_counts,
                color_continuous_scale="blues"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Total Fleet Composition
            if ev_total > 0 and non_ev_total > 0:
                fleet_data = pd.DataFrame({
                    'Vehicle Type': ['Electric Vehicles', 'Non-Electric Vehicles'],
                    'Count': [ev_total, non_ev_total]
                })
                
                fig = px.pie(
                    fleet_data, 
                    values='Count', 
                    names='Vehicle Type',
                    title="Total Vehicle Fleet Composition"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    # Top and bottom counties analysis (from code 2)
    st.markdown("### 🏆 County Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Top 5 Counties by EV Total")
        top_counties = df.groupby('County')['Electric Vehicle (EV) Total'].sum().sort_values(ascending=False).head(5)
        for i, (county, total) in enumerate(top_counties.items(), 1):
            st.write(f"{i}. **{county}**: {total:,.0f} EVs")
    
    with col2:
        st.markdown("#### Bottom 5 Counties by EV Total")
        bottom_counties = df.groupby('County')['Electric Vehicle (EV) Total'].sum().sort_values().head(5)
        for i, (county, total) in enumerate(bottom_counties.items(), 1):
            st.write(f"{i}. **{county}**: {total:,.0f} EVs")
    
    # Feature engineering insights
    st.markdown("### ⚙️ Feature Engineering Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Lag Features Distribution")
        if 'ev_total_lag1' in df.columns:
            fig = px.histogram(
                df, 
                x='ev_total_lag1', 
                title="Distribution of 1-Month Lag Features",
                labels={'ev_total_lag1': 'EV Total (Previous Month)'}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Growth Rate Distribution")
        if 'ev_total_pct_change_1' in df.columns:
            # Remove extreme outliers for visualization
            clean_pct_change = df['ev_total_pct_change_1']
            clean_pct_change = clean_pct_change[(clean_pct_change >= -1) & (clean_pct_change <= 2)]
            
            fig = px.histogram(
                x=clean_pct_change, 
                title="Distribution of Month-over-Month Growth Rates",
                labels={'x': 'Growth Rate (%)'}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

# Tab 5: Model Information (Enhanced with code 2 details)
with tab5:
    st.header("ℹ️ Advanced Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🤖 Model Architecture")
        st.info(f"""
        **Algorithm:** Random Forest Regressor  
        **Training Method:** Time-aware train-test split  
        **Hyperparameter Tuning:** RandomizedSearchCV  
        **Cross-validation:** 3-fold CV  
        **Training Data:** {len(df):,} records  
        **Features Used:** {len(features)} engineered features  
        **Counties Covered:** {len(county_list)}  
        **Date Range:** {df['Date'].min().strftime('%Y-%m')} to {df['Date'].max().strftime('%Y-%m')}
        """)
        
        st.markdown("### 📋 Feature Engineering")
        feature_descriptions = {
            "months_since_start": "Time progression indicator",
            "county_encoded": "Geographic location encoding",
            "ev_total_lag1": "EV count from 1 month ago",
            "ev_total_lag2": "EV count from 2 months ago", 
            "ev_total_lag3": "EV count from 3 months ago",
            "ev_total_roll_mean_3": "3-month rolling average",
            "ev_total_pct_change_1": "1-month growth rate",
            "ev_total_pct_change_3": "3-month growth rate",
            "ev_growth_slope": "6-month growth trend slope"
        }
        
        for feature, description in feature_descriptions.items():
            st.markdown(f"• **{feature}**: {description}")
    
    with col2:
        st.markdown("### 🎯 Model Performance & Methodology")
        st.success("""
        **Advanced Features:**
        - ✅ Outlier detection and winsorization
        - ✅ Time series lag features (1-3 months)
        - ✅ Rolling statistical features
        - ✅ Percent change calculations
        - ✅ Growth trend analysis
        - ✅ County-specific encoding
        - ✅ Cumulative growth patterns
        - ✅ Polynomial trend fitting
        """)
        
        st.markdown("### 🔧 Data Preprocessing")
        st.info("""
        **Quality Assurance:**
        - Outlier capping using IQR method
        - Missing value imputation
        - Infinite value cleaning
        - Time-aware feature creation
        - Categorical encoding
        - Feature scaling compatibility
        """)
        
        st.markdown("### ⚠️ Model Limitations")
        st.warning("""
        **Important Considerations:**
        - Forecasts based on historical patterns
        - External factors may affect accuracy
        - Policy changes not incorporated
        - Economic conditions not modeled
        - Technology adoption curves may shift
        - Use as guidance, not absolute predictions
        """)
    
    # Model hyperparameters (if available)
    st.markdown("### ⚙️ Model Configuration")
    
    try:
        # Try to get model parameters
        model_params = model.get_params()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Tree Parameters")
            st.code(f"""
n_estimators: {model_params.get('n_estimators', 'N/A')}
max_depth: {model_params.get('max_depth', 'N/A')}
min_samples_split: {model_params.get('min_samples_split', 'N/A')}
            """)
        
        with col2:
            st.markdown("#### Sampling Parameters")
            st.code(f"""
min_samples_leaf: {model_params.get('min_samples_leaf', 'N/A')}
max_features: {model_params.get('max_features', 'N/A')}
bootstrap: {model_params.get('bootstrap', 'N/A')}
            """)
        
        with col3:
            st.markdown("#### Other Parameters")
            st.code(f"""
random_state: {model_params.get('random_state', 'N/A')}
n_jobs: {model_params.get('n_jobs', 'N/A')}
oob_score: {model_params.get('oob_score', 'N/A')}
            """)
    
    except Exception as e:
        st.warning("Model parameters not available for display")
    
    # Feature importance visualization
    st.markdown("### 📊 Feature Importance Analysis")
    
    try:
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(
            feature_importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Feature Importance Ranking',
            labels={'Importance': 'Importance Score', 'Feature': 'Features'}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Top 3 most important features
        st.markdown("#### 🌟 Top 3 Most Important Features")
        for i, (_, row) in enumerate(feature_importance_df.head(3).iterrows(), 1):
            st.write(f"{i}. **{row['Feature']}**: {row['Importance']:.4f}")
    
    except Exception as e:
        st.warning("Feature importance analysis not available")
    
    # File status
    st.markdown("### 📁 File Status & Dependencies")
    
    files_status = []
    files_to_check = [
        ("Model File", MODEL_PATH),
        ("Label Encoder", ENCODER_PATH),
        ("Data File", DATA_PATH)
    ]
    
    for name, path in files_to_check:
        try:
            import os
            if os.path.exists(path):
                size = os.path.getsize(path)
                files_status.append({"File": name, "Status": "✅ Found", "Size": f"{size/1024/1024:.1f} MB"})
            else:
                files_status.append({"File": name, "Status": "❌ Missing", "Size": "N/A"})
        except:
            files_status.append({"File": name, "Status": "❓ Unknown", "Size": "N/A"})
    
    status_df = pd.DataFrame(files_status)
    st.dataframe(status_df, use_container_width=True)
    
    # Model training instructions
    st.markdown("### 🚀 Model Training Pipeline")
    st.code("""
# Complete pipeline from code 2:
1. Data Loading & Preprocessing
2. Outlier Detection & Treatment
3. Feature Engineering (9 features)
4. Time-aware Train-Test Split
5. RandomizedSearchCV Hyperparameter Tuning
6. Model Training & Evaluation
7. Model Persistence & Validation
8. Multi-county Forecasting
    """, language="python")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 20px;'>"
    "🚗 Advanced EV Adoption Forecasting Dashboard | Enhanced with ML Pipeline from Code 2<br>"
    "Built with Streamlit, Random Forest & Advanced Feature Engineering"
    "</div>", 
    unsafe_allow_html=True
)