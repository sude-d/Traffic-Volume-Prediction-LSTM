import streamlit as st
import requests
st.title("Traffic Volume Prediction")
st.write(
    "Enter the following parameters to predict traffic volume:"
)
st.markdown("""
- **Temperature (K)**: The temperature in Kelvin.
- **Rain (mm)**: The amount of rainfall in millimeters.
- **Snow (mm)**: The amount of snowfall in millimeters.
- **Clouds (%)**: The cloud cover percentage.
- **Hour (0-23)**: The hour of the day.
- **Day of Week (0-6)**: The day of the week.
- **Month (1-12)**: The month of the year.
""")
sequence_input = []
with st.form("manual_input_form"):
    st.subheader("Input Parameters for 24 Hours")
    for i in range(24):
        st.markdown(f"### Hour {i}")
        temp = st.number_input(
            f"Temperature (K) for Hour {i}",
            min_value=0.0,
            max_value=400.0,
            value=290.0,
            step=0.1
        )
        rain = st.number_input(
            f"Rain (mm) for Hour {i}",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=0.1
        )
        snow = st.number_input(
            f"Snow (mm) for Hour {i}",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=0.1
        )
        clouds = st.number_input(
            f"Clouds (%) for Hour {i}",
            min_value=0,
            max_value=100,
            value=50,
            step=1
        )
        day_of_week = st.number_input(
            f"Day of Week (0-6) for Hour {i}",
            min_value=0,
            max_value=6,
            value=i % 7,
            step=1
        )
        month = st.number_input(
            f"Month (1-12) for Hour {i}",
            min_value=1,
            max_value=12,
            value=(i % 12) + 1,
            step=1
        )
        sequence_input.append([
            temp,
            rain,
            snow,
            clouds,
            i,
            day_of_week,
            month
        ])
    submitted = st.form_submit_button(
        "Predict Traffic Volume"
    )
if submitted:
    payload = {
        "sequence": sequence_input
    }
    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=payload
        )
        if response.status_code == 200:
            result = response.json()
            st.success("Prediction completed successfully!")
            st.subheader("Predicted Traffic Volume")
            if isinstance(result, dict):
                if "prediction" in result:
                    st.metric(
                        "Traffic Volume",
                        result["prediction"]
                    )
                else:
                    st.write(result)
            else:
                st.write(result)
        else:
            st.error(
                f"API Error: {response.status_code}"
            )
            st.write(response.text)
    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to the prediction API. "
            "Make sure the FastAPI server is running."
        )
    except Exception as e:
        st.error(
            f"An unexpected error occurred: {e}"
        )