import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# sqlite connection
db_path = "kafka_data.db"
conn = sqlite3.connect(db_path)

#Clean up naming convention
display_to_column = {
    "Mean Temperature": "temp_mean__temp_moyenne",
    "Max Temperature": "temp_max__temp_max",
    "Min Temperature": "temp_min__temp_min",
    "Total Precip": "total_precip__precip_totale",
    "Rain": "rain_mm",
    "Snow": "snow_mm",
    "Sea Level Pressure": "pressure_sea_level_hpa",
    "Station Pressure": "pressure_station_hpa",
    "Wind Speed": "wind_speed_kph"
}

numeric_cols = list(display_to_column.values())

#create seasons 
seasons = {
    "Winter": [12, 1, 2],
    "Spring": [3, 4, 5],
    "Summer": [6, 7, 8],
    "Autumn": [9, 10, 11]
}


st.title("Weather Dashboard")

#city selection
preview_df = pd.read_sql_query("SELECT station_name FROM processed_data;", conn)
cities = sorted(preview_df["station_name"].unique())
selected_city = st.sidebar.selectbox("City:", cities)


#Toggle between Single or multivarible analysis 
analysis_mode = st.sidebar.radio(
    "Select Analysis Type:",
    ["Single Variable Analysis", "Correlation Analysis"]
)

#Month/Season filter
filter_type = st.sidebar.radio("Filter by:", ["None", "Month", "Season"])
selected_months = []
selected_season = None

if filter_type == "Month":
    selected_months = st.sidebar.multiselect("Select Month(s):", list(range(1, 13)))
elif filter_type == "Season":
    selected_season = st.sidebar.selectbox("Select Season:", list(seasons.keys()))
    selected_months = seasons[selected_season]


#Single variable analysis
if analysis_mode == "Single Variable Analysis":
    selected_column_display = st.sidebar.selectbox("Variable:", list(display_to_column.keys()))
    selected_column = display_to_column[selected_column_display]

    selected_operation = st.sidebar.selectbox(
        "Operation:",
        ["Descriptive Statistics", "Time Series Analysis", "Extreme Events / Anomalies"]
    )

    st.write(f"Selected City: **{selected_city}**")
    st.write(f"Selected Variable: **{selected_column_display}**")
    if filter_type != "None":
        st.write(f"Filter: **{filter_type} - {selected_months}**")
    st.write(f"Selected Operation: **{selected_operation}**")

    #Single variable column query. Ensure we are not pulling rows with missing values 
    query = f"""
    SELECT station_name, year, month, {selected_column}
    FROM processed_data
    WHERE station_name = '{selected_city}'
    AND year IS NOT NULL AND year <> ''
    AND month IS NOT NULL AND month <> ''
    AND {selected_column} IS NOT NULL AND {selected_column} <> ''
    """
    city_data = pd.read_sql_query(query, conn)

    # Apply month/season filter
    if selected_months:
        city_data = city_data[city_data["month"].isin(selected_months)]

    #slider for year range
    available_years = sorted(city_data["year"].astype(int).unique())
    if available_years:
        selected_year_range = st.sidebar.select_slider(
            "Select Year Range:",
            options=available_years,
            value=(available_years[0], available_years[-1])
        )
        city_data = city_data[
            (city_data["year"].astype(int) >= selected_year_range[0]) &
            (city_data["year"].astype(int) <= selected_year_range[1])
        ]

    if city_data.empty:
        st.warning("No data available for the selected city and filter.")
        st.stop()

    # Rename column for display
    city_data.rename(columns={selected_column: selected_column_display}, inplace=True)

    # Time column needed for time analysis 
    city_data["time"] = pd.to_datetime(
        city_data["year"].astype(str) + "-" + city_data["month"].astype(str).str.zfill(2),
        format="%Y-%m",
        errors="coerce"
    )

    #Data Preview section 
    st.subheader("Data Preview")
    preview_no_time = city_data.drop(columns=["time"])
    st.dataframe(preview_no_time)

    #Descriptive Statistics section
    if selected_operation == "Descriptive Statistics":
        st.subheader("Descriptive Statistics")
        st.write(city_data[selected_column_display].describe())

    #Time Series plot
    if selected_operation == "Time Series Analysis":
        st.subheader("Time Series Plot")
        st.line_chart(city_data.set_index("time")[selected_column_display])

    #Data Outliers 
    if selected_operation == "Extreme Events / Anomalies":
        st.subheader("Extreme Events / Anomalies")
        threshold = st.number_input(f"Threshold for {selected_column_display}", value=0.0)
        extreme_events = city_data[city_data[selected_column_display] > threshold]
        preview_no_time = extreme_events.drop(columns=["time"])
        st.dataframe(preview_no_time)
        # st.dataframe(extreme_events)


#Correlation analysis section
elif analysis_mode == "Correlation Analysis":
    selected_columns_display = st.sidebar.multiselect(
        "Select Variables for Correlation",
        list(display_to_column.keys()),
        default=list(display_to_column.keys())
    )
    selected_columns = [display_to_column[c] for c in selected_columns_display]

    st.write(f"Selected City: **{selected_city}**")
    if filter_type != "None":
        st.write(f"Filter: **{filter_type} - {selected_months}**")
    st.write(f"Variables: **{selected_columns_display}**")

    if not selected_columns:
        st.warning("Select at least one variable for correlation.")
        st.stop()

    #Query for multivariable analysis, the previous query above only looks at one variable.
    columns_str = ", ".join(["station_name", "year", "month"] + selected_columns)
    not_null_conditions = " AND ".join([f"{col} IS NOT NULL AND {col} <> ''" for col in selected_columns])

    corr_query = f"""
    SELECT {columns_str}
    FROM processed_data
    WHERE station_name = '{selected_city}'
    AND year IS NOT NULL AND year <> ''
    AND month IS NOT NULL AND month <> ''
    AND {not_null_conditions}
    ORDER BY year, month
    """
    corr_data = pd.read_sql_query(corr_query, conn)

    # Apply month/season filter
    if selected_months:
        corr_data = corr_data[corr_data["month"].isin(selected_months)]

    # Apply year slider
    available_years_corr = sorted(corr_data["year"].astype(int).unique())
    if available_years_corr:
        selected_year_range_corr = st.sidebar.select_slider(
            "Select Year Range for Correlation:",
            options=available_years_corr,
            value=(available_years_corr[0], available_years_corr[-1])
        )
        corr_data = corr_data[
            (corr_data["year"].astype(int) >= selected_year_range_corr[0]) &
            (corr_data["year"].astype(int) <= selected_year_range_corr[1])
        ]

    if corr_data.empty or len(corr_data) < 2:
        st.warning("Not enough data for correlation after applying filters.")
        st.stop()


    #Correlation Matrix
    st.subheader("Correlation Matrix")
    corr_matrix = corr_data[selected_columns].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    #Scatterplot with Regression Line
    st.subheader("X vs Y Scatterplot with Regression Line")

    #Two variables (X and Y)
    if len(selected_columns_display) < 2:
        st.info("Select at least two variables above to enable scatterplot regression.")
    else:
        x_var_display = st.selectbox("Select X Variable:", selected_columns_display)
        y_var_display = st.selectbox("Select Y Variable:", selected_columns_display, index=1)

        x_var = display_to_column[x_var_display]
        y_var = display_to_column[y_var_display]

        # Ensure usable rows only
        scatter_df = corr_data[[x_var, y_var]].dropna()

        if scatter_df.empty:
            st.warning("No usable data available for the selected X and Y variables.")
        else:
            # Compute regression using sklearn
            model = LinearRegression()
            model.fit(scatter_df[[x_var]], scatter_df[y_var])

            # Predictions for regression line
            scatter_df["pred"] = model.predict(scatter_df[[x_var]])

            # Plotting
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(scatter_df[x_var], scatter_df[y_var], alpha=0.7)
            ax.plot(scatter_df[x_var], scatter_df["pred"], linewidth=2)

            ax.set_xlabel(x_var_display)
            ax.set_ylabel(y_var_display)
            ax.set_title(f"{x_var_display} vs {y_var_display} with Regression Line")

            st.pyplot(fig)


    #Multiple Linear Regression
    st.subheader("Multiple Linear Regression")

    st.markdown("Select a target variable (Y) and one or more predictors (X):")

    #Select target (Y)
    target_display = st.selectbox(
        "Select Target (Y):",
        selected_columns_display
    )
    target_col = display_to_column[target_display]

    #Select predictors (X)
    predictor_display = st.multiselect(
        "Select Predictor Variables (X):",
        [v for v in selected_columns_display if v != target_display]
    )

    if predictor_display:
        predictor_cols = [display_to_column[p] for p in predictor_display]

        # Ensure usable rows only
        mlr_df = corr_data[[target_col] + predictor_cols].dropna()

        if len(mlr_df) < 5:
            st.warning("Not enough data for regression after filtering.")
        else:
            X = mlr_df[predictor_cols]
            y = mlr_df[target_col]

            # Fit model
            model = LinearRegression()
            model.fit(X, y)

            # Predictions & R2
            predictions = model.predict(X)
            r2 = model.score(X, y)

            st.markdown("### Regression Results")
            st.write(f"**Target:** {target_display}")
            st.write(f"**Predictors:** {predictor_display}")

            #Display coefficients
            coef_table = pd.DataFrame({
                "Predictor": predictor_display,
                "Coefficient": model.coef_
            })

            st.write("#### Coefficients")
            st.dataframe(coef_table)

            st.write(f"**Intercept:** {model.intercept_:.4f}")
            st.write(f"**R² Score:** {r2:.4f}")

            #Predictions table
            st.write("#### Predicted vs Actual")
            pred_df = pd.DataFrame({
                "Actual": y,
                "Predicted": predictions
            })
            st.dataframe(pred_df.head(20))

            #Residual Plot
            residuals = y - predictions
            fig_res, ax_res = plt.subplots(figsize=(8, 5))
            ax_res.scatter(predictions, residuals, alpha=0.7)
            ax_res.axhline(0, linestyle="--")
            ax_res.set_xlabel("Predicted Values")
            ax_res.set_ylabel("Residuals")
            ax_res.set_title("Residual Plot")
            st.pyplot(fig_res)

    else:
        st.info("Select at least one predictor variable to run multiple linear regression.")

pd.show_versions()