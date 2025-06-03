import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import math
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import plotly.express as px
from dotenv import load_dotenv

# Set page configuration
st.set_page_config(layout="wide", page_title="GDD Calculator")

# Initialize session state for all inputs
def initialize_session_state():
    """Initialize all session state variables with default values."""
    defaults = {
        'reset_clicked': False,
        'persisted_t_base': 50.0,
        'persisted_t_lower': 50.0,
        'persisted_t_upper': 86.0,
        'default_start_date': None,
        'default_end_date': None,
        'default_harvest_date': None,
        'city': "",
        'state': "",
        'method': "average",
        'show_summary': False,
        'calculation_done': False,
        'df_calc': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Call the initialization function at the start
initialize_session_state()

# Inject custom CSS to style the UI
st.markdown(
    """
    <style>
    /* Hide the entire header bar */
    [data-testid="stHeader"] {
        display: none;
    }
    /* Hide the Streamlit watermark/footer */
    [data-testid="stDecoration"] {
        display: none;
    }
    /* Adjust padding */
    [data-testid="stAppViewContainer"] {
        padding-top: 0px !important;
        padding-left: 10px !important;
        padding-right: 10px !important;
    }
    /* Style the sidebar */
    .css-1d391kg {
        background-color: #f5f5f5;
        padding: 20px;
    }
    /* Style sidebar labels */
    .css-1v0mbdj {
        font-weight: bold;
        margin-bottom: 5px;
    }
    /* Style input boxes */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stDateInput > div > div > input,
    .stSelectbox > div > div > div {
        border: 1px solid #ccc;
        border-radius: 5px;
        padding: 5px;
    }
    /* Style buttons */
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 8px 16px;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    /* Style the main chart area */
    .css-1kyxreq {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    /* Style the custom buttons for sections */
    .custom-button-container {
        display: flex;
        justify-content: center;
        gap: 10px;
        margin-top: 20px;
    }
    /* Style the header */
    h1, h2 {
        text-align: center;
        color: #333;
    }
    h3 {
        text-align: center;
        color: #666;
        font-size: 16px;
        margin-top: -10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def reset_inputs():
    """Reset all input values in session state"""
    st.session_state['city'] = ""
    st.session_state['state'] = ""
    st.session_state['default_start_date'] = None
    st.session_state['default_end_date'] = None
    st.session_state['default_harvest_date'] = None
    st.session_state['persisted_t_base'] = 50.0
    st.session_state['persisted_t_lower'] = 50.0
    st.session_state['persisted_t_upper'] = 86.0
    st.session_state['method'] = "average"
    st.session_state['show_summary'] = False
    st.session_state['calculation_done'] = False
    st.session_state['df_calc'] = None

# Function to Fetch Weather Data (Simulated)
def fetch_weather_data_from_api(api_key, city, state, start_date, end_date,
                               unit_group="us", elements="datetime,tempmin,tempmax"):
    """
    Simulates weather data between start_date and end_date for a given city, state.
    Returns a DataFrame with columns ['datetime','tempmin','tempmax',...].
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    avg_temps = np.linspace(40, 70, len(dates))
    df = pd.DataFrame({
        'datetime': dates,
        'tempmin': avg_temps - 10 + np.random.uniform(-5, 5, len(dates)),
        'tempmax': avg_temps + 10 + np.random.uniform(-5, 5, len(dates))
    })
    return df

# Simulated function for drying weather data
def fetch_drying_weather_data_from_api(api_key, city, state, start_date, end_date, unit_group="us"):
    """
    Simulates daily and hourly weather data for drying calculations.
    Expects start_date and end_date as datetime.date or datetime.datetime objects.
    Returns daily_df and hourly_df.
    """
    # Ensure start_date and end_date are datetime objects
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)

    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    daily_df = pd.DataFrame({
        'datetime': dates,
        'temp': np.random.uniform(60, 80, len(dates)),
        'dew': np.random.uniform(50, 70, len(dates)),
        'soilmoisturevol01': np.random.uniform(0.1, 0.4, len(dates))
    })
    hours = pd.date_range(start=start_date, end=end_date + timedelta(days=1), freq='H')[:-1]
    hourly_df = pd.DataFrame({
        'datetime': hours,
        'solarradiation': np.random.uniform(100, 800, len(hours))
    })
    return daily_df, hourly_df

# GDD Calculation Functions
def calc_average_gdd(t_min, t_max, T_base):
    return max(((t_max + t_min) / 2) - T_base, 0)

def calc_single_sine_gdd(t_min, t_max, T_base):
    T_mean = (t_max + t_min) / 2
    A = (t_max - t_min) / 2
    if t_max <= T_base:
        return 0.0
    if t_min >= T_base:
        return (T_mean - T_base)
    alpha = (T_base - T_mean) / A
    theta = math.acos(alpha)
    dd = ((T_mean - T_base)*(math.pi - 2*theta) + A*math.sin(2*theta)) / math.pi
    return dd

def calc_single_triangle_gdd(t_min, t_max, T_base):
    if t_max <= T_base:
        return 0.0
    if t_min >= T_base:
        return ((t_max + t_min)/2 - T_base)
    proportion_of_day = (t_max - T_base) / (t_max - t_min)
    avg_above = ((t_max + T_base) / 2) - T_base
    dd = proportion_of_day * avg_above
    return max(dd, 0)

def calc_double_sine_gdd(t_min_today, t_max_today, t_min_tomorrow, T_base):
    seg1 = calc_single_sine_gdd(t_min_today, t_max_today, T_base) * 0.5
    seg2 = calc_single_sine_gdd(t_min_tomorrow, t_max_today, T_base) * 0.5
    return seg1 + seg2

def calc_double_triangle_gdd(t_min_today, t_max_today, t_min_tomorrow, T_base):
    seg1 = calc_single_triangle_gdd(t_min_today, t_max_today, T_base) * 0.5
    seg2 = calc_single_triangle_gdd(t_min_tomorrow, t_max_today, T_base) * 0.5
    return seg1 + seg2

def calculate_daily_gdd(row, df, method="average", T_base=50.0, T_lower=50.0, T_upper=86.0):
    t_max = min(row['tmax'], T_upper)
    t_min = max(row['tmin'], T_lower)
    
    if method == "average":
        return calc_average_gdd(t_min, t_max, T_base)
    elif method == "sine":
        return calc_single_sine_gdd(t_min, t_max, T_base)
    elif method == "triangle":
        return calc_single_triangle_gdd(t_min, t_max, T_base)
    elif method in ["double_sine", "double_triangle"]:
        current_idx = row.name
        if current_idx < len(df) - 1:
            t_min_tomorrow = max(df.iloc[current_idx + 1]['tmin'], T_lower)
        else:
            t_min_tomorrow = t_min
        
        if method == "double_sine":
            return calc_double_sine_gdd(t_min, t_max, t_min_tomorrow, T_base)
        else:
            return calc_double_triangle_gdd(t_min, t_max, t_min_tomorrow, T_base)

def calculate_gdds_for_df(df, start_date, end_date, method="average", T_base=50.0, T_lower=50.0, T_upper=86.0):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df['daily_gdd'] = df.apply(
        lambda row: calculate_daily_gdd(row, df, method=method, T_base=T_base, T_lower=T_lower, T_upper=T_upper),
        axis=1
    )
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    df.loc[df['date'] < start_date, 'daily_gdd'] = 0
    df.loc[df['date'] > end_date, 'daily_gdd'] = 0
    df['cumulative_gdd'] = df['daily_gdd'].cumsum()
    return df

# Drying Calculation Functions
def merge_dfs(daily_df, hourly_df):
    '''
    Extracts the peak solar radiation for each day from the
    hourly df and adds it to the daily df.
    '''
    d_df = daily_df
    h_df = hourly_df

    # Step 1: Convert datetime columns to datetime objects
    h_df['datetime'] = pd.to_datetime(h_df['datetime'])
    d_df['datetime'] = pd.to_datetime(d_df['datetime'])

    # Step 2: Extract date from hourly timestamps
    h_df['date'] = h_df['datetime'].dt.date

    # Step 3: Group by date and get max solar radiation
    daily_peaks = h_df.groupby('date')['solarradiation'].max().reset_index()
    daily_peaks.rename(columns={'solarradiation': 'peak_solarradiation'}, inplace=True)

    # Step 4: Merge with daily dataframe
    d_df['date'] = d_df['datetime'].dt.date
    merged_df = pd.merge(d_df, daily_peaks, on='date', how='left')

    # Drop the 'date' column
    merged_df.drop(columns='date', inplace=True)

    return merged_df

def calculate_vapor_pressure_deficit(df):
    '''Takes weather df (with temp and dew in °F) and adds vapor pressure deficit column in kPa'''
    df['temp_C'] = (df['temp'] - 32) / 1.8
    df['dew_C'] = (df['dew'] - 32) / 1.8
    df['saturation_vapor_pressure'] = 0.6108 * np.exp((17.27 * df['temp_C']) / (df['temp_C'] + 237.3))
    df['actual_vapor_pressure'] = 0.6108 * np.exp((17.27 * df['dew_C']) / (df['dew_C'] + 237.3))
    df['vapor_pressure_deficit'] = df['saturation_vapor_pressure'] - df['actual_vapor_pressure']
    return df

def swath_density_conversion(plants_per_sqft, g_per_plant=25):
    '''Convert swath density from plants/ft^2 to g/m^2'''
    plants_per_sqm = plants_per_sqft * 10.764
    return plants_per_sqm * g_per_plant

def calculate_drying_rate_constant(SI, VPD, DAY, SM, SD, AR=0):
    '''
    Calculate drying rate constant based on inputs.
    '''
    drying_rate = ((SI * (1. + 9.03*AR)) + (43.8 * VPD)) / ((61.4 * SM) + SD * (1.82 - 0.83 * DAY) * ((1.68 + 24.8 * AR)) + 2767)
    return drying_rate

def predict_moisture_content(df, startdate, swath_density=450, starting_moisture=0.80, application_rate=0):
    """
    Simulate daily drying and return DataFrame with moisture content predictions.
    """
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df[df['datetime'] >= pd.to_datetime(startdate)].copy()
    df.sort_values('datetime', inplace=True)

    df = calculate_vapor_pressure_deficit(df)

    moisture_contents = [starting_moisture]
    drying_rates = []
    current_moisture = starting_moisture

    for idx, row in df.iterrows():
        day_number = len(moisture_contents) - 1
        DAY = 1 if day_number == 0 else 0
        SI = row['peak_solarradiation']
        VPD = row['vapor_pressure_deficit']
        SM = 100 * row['soilmoisturevol01'] if not np.isnan(row['soilmoisturevol01']) else 10
        SD = swath_density
        AR = application_rate

        k = calculate_drying_rate_constant(SI, VPD, DAY, SM, SD, AR)
        current_moisture *= np.exp(-k)
        moisture_contents.append(current_moisture)
        drying_rates.append(k)

        if current_moisture <= 0.08:
            break

    result_df = df.iloc[:len(moisture_contents)-1].copy()
    result_df['drying_rates'] = drying_rates
    result_df['predicted_moisture'] = moisture_contents[:-1]
    result_df['predicted_moisture_pct'] = result_df['predicted_moisture'] * 100
    result_df = result_df.dropna(subset=['predicted_moisture'])
    return result_df

# Simulate historical data for the chart
def simulate_historical_data(df):
    dates = df['date']
    cumulative_gdd = df['cumulative_gdd']
    gdd_15yr_avg = cumulative_gdd * 0.95
    gdd_30yr_normal = cumulative_gdd * 1.05
    gdd_period_min = cumulative_gdd * 0.8
    gdd_period_max = cumulative_gdd * 1.2
    historical_df = pd.DataFrame({
        'date': dates,
        'cumulative_gdd': cumulative_gdd,
        '15yr_avg': gdd_15yr_avg,
        '30yr_normal': gdd_30yr_normal,
        'period_min': gdd_period_min,
        'period_max': gdd_period_max,
        'daily_gdd': df['daily_gdd'] if 'daily_gdd' in df.columns else [0] * len(dates)
    })
    return historical_df

# Main Application
def main():
    load_dotenv()
    API_KEY = os.getenv("API_KEY")
    
    title_container = st.container()
    with title_container:
        st.markdown("<h1 style='text-align: center;'>zalliant-Integrating Forecasting for Weather-Optimized Crop Cutting</h1>", unsafe_allow_html=True)

    # Create sidebar for inputs
    with st.sidebar:
        st.write("### Input Parameters")

        algorithm_options = ["GDD Calculation", "Drying Calculation"]
        algorithm = st.selectbox(
            "Select Calculation Algorithm",
            options=algorithm_options,
            key="algorithm"
        )

        # Common inputs
        st.write("### Location")
        city = st.text_input("City", value=st.session_state['city'], placeholder="Enter city", key="city_input")
        state = st.text_input("State", value=st.session_state['state'], placeholder="Enter state", key="state_input")
        st.session_state['city'] = city
        st.session_state['state'] = state

        if algorithm == "GDD Calculation":
            st.write("### GDD Parameters")
            start_date = st.date_input(
                "Start Date",
                value=st.session_state['default_start_date'],
                key="gdd_start_date"
            )
            end_date = st.date_input(
                "End Date",
                value=st.session_state['default_end_date'],
                key="gdd_end_date"
            )
            st.session_state['default_start_date'] = start_date
            st.session_state['default_end_date'] = end_date

            method_options = ["average", "sine", "triangle", "double_sine", "double_triangle"]
            method = st.selectbox(
                "Method",
                options=method_options,
                index=method_options.index(st.session_state['method']),
                key="method_selectbox"
            )
            st.session_state['method'] = method

            t_base = st.number_input(
                "T_base",
                min_value=0.0,
                max_value=100.0,
                value=st.session_state['persisted_t_base'],
                step=0.1,
                key="t_base"
            )
            t_lower = st.number_input(
                "T_lower",
                min_value=0.0,
                max_value=100.0,
                value=st.session_state['persisted_t_lower'],
                step=0.1,
                key="t_lower"
            )
            t_upper = st.number_input(
                "T_upper",
                min_value=0.0,
                max_value=100.0,
                value=st.session_state['persisted_t_upper'],
                step=0.1,
                key="t_upper"
            )

            st.markdown("### GDD Target")
            col1, col2 = st.columns([1, 3])
            with col1:
                use_gdd_target = st.checkbox("Set Target", value=False, key="use_gdd_target")
            with col2:
                gdd_target = st.number_input(
                    "GDD Target",
                    value=755.0,
                    step=1.0,
                    disabled=not use_gdd_target,
                    key="gdd_target"
                )

            calculate_button = st.button("Calculate GDD", key="gdd_calculate")

        elif algorithm == "Drying Calculation":
            st.write("### Drying Parameters")
            harvest_date = st.date_input(
                "Harvest Date",
                value=st.session_state['default_harvest_date'],
                key="harvest_date"
            )
            st.session_state['default_harvest_date'] = harvest_date

            drying_button = st.button("Estimate Drying", key="drying_calculate")

        # Reset button
        st.markdown("---")
        if st.button("Reset Inputs"):
            reset_inputs()
            st.rerun()

    # Main content area
    if algorithm == "GDD Calculation" and 'gdd_calculate' in st.session_state and st.session_state['gdd_calculate']:
        if not city or not state:
            st.error("Please enter both City and State")
        elif st.session_state['default_start_date'] is None or st.session_state['default_end_date'] is None:
            st.error("Please select both Start Date and End Date")
        elif st.session_state['default_start_date'] >= st.session_state['default_end_date']:
            st.error("End Date must be after Start Date")
        else:
            with st.spinner("Fetching weather data and calculating GDD..."):
                try:
                    df_api = fetch_weather_data_from_api(
                        api_key=API_KEY,
                        city=city,
                        state=state,
                        start_date=st.session_state['default_start_date'],
                        end_date=st.session_state['default_end_date'],
                        unit_group="us",
                        elements="datetime,tempmin,tempmax"
                    )

                    if not df_api.empty:
                        df_renamed = df_api.copy()
                        df_renamed.rename(columns={
                            "datetime": "date",
                            "tempmax": "tmax",
                            "tempmin": "tmin"
                        }, inplace=True)

                        df_calc = calculate_gdds_for_df(
                            df=df_renamed,
                            start_date=st.session_state['default_start_date'],
                            end_date=st.session_state['default_end_date'],
                            method=st.session_state['method'],
                            T_base=t_base,
                            T_lower=t_lower,
                            T_upper=t_upper
                        )

                        df_calc = simulate_historical_data(df_calc)
                        
                        st.session_state['df_calc'] = df_calc
                        st.session_state['calculation_done'] = True
                        target_date = None
                        if use_gdd_target:
                            target_row = df_calc[df_calc['cumulative_gdd'] >= gdd_target].iloc[0] if any(df_calc['cumulative_gdd'] >= gdd_target) else None
                            if target_row is not None:
                                target_date = target_row['date']
                        st.session_state['calc_params'] = {
                            'city': city,
                            'state': state,
                            'start_date': st.session_state['default_start_date'],
                            'end_date': st.session_state['default_end_date'],
                            't_base': t_base,
                            'gdd_target': gdd_target,
                            'target_date': target_date
                        }

                        current_date = pd.to_datetime("2025-05-30")
                        season_to_date = df_calc[df_calc['date'] <= current_date]
                        forecast = df_calc[(df_calc['date'] > current_date) & (df_calc['date'] <= current_date + pd.Timedelta(days=6))]

                        location_str = f"{city}, {state}"
                        st.markdown(f"<h2 style='text-align: left;'>2025 Cumulative Growing Degree Days (Base {int(t_base)})</h2>", unsafe_allow_html=True)
                        st.markdown(f"<h1 style='text-align: left;'>Location: {location_str}</h1>", unsafe_allow_html=True)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=season_to_date['date'],
                            y=season_to_date['cumulative_gdd'],
                            name="Season to Date",
                            line=dict(color="#00FF00", width=2)
                        ))
                        fig.add_trace(go.Scatter(
                            x=forecast['date'],
                            y=forecast['cumulative_gdd'],
                            name="6 Day Forecast",
                            line=dict(color="#FF0000", width=2)
                        ))
                        fig.add_trace(go.Scatter(
                            x=df_calc['date'],
                            y=df_calc['15yr_avg'],
                            name="15 Year Average",
                            line=dict(color="#0000FF", width=2)
                        ))
                        fig.add_trace(go.Scatter(
                            x=df_calc['date'],
                            y=df_calc['30yr_normal'],
                            name="30 Year 'Normal'",
                            line=dict(color="#800080", width=2)
                        ))
                        fig.add_trace(go.Scatter(
                            x=df_calc['date'],
                            y=df_calc['period_max'],
                            name="Period of Record",
                            line=dict(color="gray", width=0),
                            showlegend=False,
                            mode='lines',
                            fillcolor='rgba(128,128,128,0.2)',
                            fill='tonexty'
                        ))
                        fig.add_trace(go.Scatter(
                            x=df_calc['date'],
                            y=df_calc['period_min'],
                            name="Period of Record",
                            line=dict(color="gray", width=0),
                            showlegend=True,
                            mode='lines',
                            fillcolor='rgba(128,128,128,0.2)',
                            fill='tonexty'
                        ))

                        if use_gdd_target:
                            if target_date is not None:
                                fig.add_shape(
                                    type="line",
                                    x0=target_date,
                                    y0=0,
                                    x1=target_date,
                                    y1=df_calc['cumulative_gdd'].max(),
                                    line=dict(
                                        color="green",
                                        width=2,
                                        dash="solid",
                                    ),
                                    name="Target Fcst"
                                )
                                fig.add_annotation(
                                    x=target_date,
                                    y=df_calc['cumulative_gdd'].max(),
                                    text=f"Target Fcst: {gdd_target}. Reached on {target_date.strftime('%Y-%m-%d')}",
                                    showarrow=True,
                                    arrowhead=1,
                                    ax=20,
                                    ay=-30,
                                    font=dict(color="white")
                                )
                            else:
                                st.warning(f"GDD Target ({gdd_target}) not reached within the date range (up to {st.session_state['default_end_date'].strftime('%B %d, %Y')}).")

                        fig.update_layout(
                            title=f"2025 Cumulative Growing Degree Days (Base {int(t_base)})",
                            xaxis_title="Date",
                            yaxis_title="Cumulative GDD",
                            height=650,
                            margin=dict(l=60, r=30, t=60, b=60),
                            plot_bgcolor="black",
                            paper_bgcolor="black",
                            font=dict(color='white'),
                            xaxis=dict(
                                title=dict(
                                    text="Date",
                                    font=dict(color='white')
                                ),
                                tickformat="%b %Y",
                                tickfont=dict(color='white'),
                                showgrid=False,
                                gridcolor='gray',
                                linecolor='white',
                                showline=True,
                                ticks='outside'
                            ),
                            yaxis=dict(
                                title=dict(
                                    text="Cumulative GDD",
                                    font=dict(color='white')
                                ),
                                tickfont=dict(color='white'),
                                showgrid=False,
                                gridcolor='gray',
                                linecolor='white',
                                showline=True,
                                ticks='outside'
                            ),
                            legend=dict(
                                font=dict(color='white'),
                                bgcolor='rgba(0,0,0,0)',
                                orientation='v',
                                x=1.01,
                                y=1
                            ),
                            hovermode="x unified",
                            hoverlabel=dict(
                                bgcolor="white",
                                font=dict(color="black")
                            )
                        )

                        fig.update_xaxes(showspikes=True, spikecolor="red", spikethickness=1, spikedash="solid")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Failed to fetch weather data. Please check your inputs and try again.")
                except Exception as e:
                    st.error(f"An error occurred while processing the data: {str(e)}")

    elif algorithm == "Drying Calculation" and 'drying_calculate' in st.session_state and st.session_state['drying_calculate']:
        if not city or not state:
            st.error("Please enter both City and State")
        elif st.session_state['default_harvest_date'] is None:
            st.error("Please select a Harvest Date")
        else:
            with st.spinner("Fetching weather data and estimating drying..."):
                try:
                    daily_df, hourly_df = fetch_drying_weather_data_from_api(
                        api_key=API_KEY,
                        city=city,
                        state=state,
                        start_date=st.session_state['default_harvest_date'],
                        end_date=st.session_state['default_harvest_date'] + relativedelta(months=2),
                        unit_group="us"
                    )

                    if not daily_df.empty and not hourly_df.empty:
                        merged_df = merge_dfs(daily_df, hourly_df)
                        drying_df = predict_moisture_content(
                            df=merged_df,
                            startdate=st.session_state['default_harvest_date']
                        )

                        location_str = f"{city}, {state}"
                        st.header(f"Crop Drying Prediction Results")
                        st.subheader(f"Location: {location_str} | Harvest Date: {st.session_state['default_harvest_date'].strftime('%Y-%m-%d')}")

                        fig = px.line(
                            drying_df,
                            x="datetime",
                            y="predicted_moisture_pct",
                            title=f"Expected Crop Moisture Content Over Time"
                        )
                        fig.update_layout(
                            xaxis_title="Date",
                            yaxis_title="Moisture Content (%)",
                            height=600
                        )

                        fig.add_shape(
                            type="line",
                            x0=drying_df["datetime"].min(),
                            y0=15,
                            x1=drying_df["datetime"].max(),
                            y1=15,
                            line=dict(
                                color="green",
                                width=2,
                                dash="dash",
                            )
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        target_moisture = 0.15
                        baling_ready_row = drying_df[drying_df['predicted_moisture'] <= target_moisture].iloc[0] if any(drying_df['predicted_moisture'] <= target_moisture) else None

                        if baling_ready_row is not None:
                            baling_date = pd.to_datetime(baling_ready_row['datetime'])
                            days_to_baling = (baling_date - pd.to_datetime(st.session_state['default_harvest_date'])).days
                            st.info(f"Based on weather forecasts, the crop will reach optimal baling moisture (15%) in approximately **{days_to_baling} days** from harvest (around {baling_date.strftime('%B %d, %Y')}).")
                        else:
                            st.warning("The crop may not reach optimal baling moisture (15%) within the forecast period.")
                    else:
                        st.error("Failed to fetch weather data. Please check your inputs and try again.")
                except Exception as e:
                    st.error(f"An error occurred while processing the data: {str(e)}")

    # Show summary table if calculation is done
    if st.session_state.get('calculation_done', False) and st.session_state.get('df_calc') is not None:
        st.markdown('<div class="custom-button-container">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Summary Table", key="summary_button"):
                st.session_state['show_summary'] = True
        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.get('show_summary', False):
            df_calc = st.session_state['df_calc']
            params = st.session_state.get('calc_params', {})
            
            current_date = pd.to_datetime("2025-05-30")
            season_to_date = df_calc[df_calc['date'] <= current_date]
            forecast = df_calc[(df_calc['date'] > current_date) & (df_calc['date'] <= current_date + pd.Timedelta(days=6))]
            
            if season_to_date.empty:
                st.warning("No data available for the season to date. Please adjust the date range to include dates before or on May 30, 2025.")
            else:
                st.markdown("### Summary Table")
                
                total_cumulative_gdd = season_to_date['cumulative_gdd'].iloc[-1] if not season_to_date.empty else 0
                days_in_season = len(season_to_date)
                avg_daily_gdd = total_cumulative_gdd / days_in_season if days_in_season > 0 else 0
                current_gdd = total_cumulative_gdd
                remaining_gdd = max(params.get('gdd_target', 755) - current_gdd, 0)

                target_date = params.get('target_date')
                if target_date is not None:
                    days_to_target = (target_date - current_date).days
                    estimated_target_date = target_date.strftime('%Y-%m-%d')
                else:
                    days_to_target = 0
                    estimated_target_date = "Not Reachable"

                summary_data = {
                    "Metric": [
                        "Start Date",
                        "End Date",
                        "Total Cumulative GDD (Season to Date)",
                        "Average Daily GDD (Season to Date)",
                        "GDD Target",
                        "Remaining GDD to Target",
                        "Estimated Days to Reach Target",
                        "Estimated Target Date",
                    ],
                    "Value": [
                        params.get('start_date', '').strftime('%Y-%m-%d') if params.get('start_date') else '',
                        params.get('end_date', '').strftime('%Y-%m-%d') if params.get('end_date') else '',
                        f"{total_cumulative_gdd:.1f}",
                        f"{avg_daily_gdd:.1f}",
                        f"{params.get('gdd_target', 755)}",
                        f"{remaining_gdd:.1f}",
                        str(days_to_target),
                        estimated_target_date,
                    ]
                }
                summary_df = pd.DataFrame(summary_data)

                st.dataframe(
                    summary_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Metric": st.column_config.TextColumn("Metric", width="medium"),
                        "Value": st.column_config.TextColumn("Value", width="medium")
                    }
                )

if __name__ == "__main__":
    main()