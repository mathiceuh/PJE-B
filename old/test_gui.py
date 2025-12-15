# streamlit_app.py

import streamlit as st
from core.algorithm_manager import manager
import csv

# ==============================
# Load Dataset (minimal, just for fitting/testing)
# ==============================
TEST_CSV_FILE = "../test.csv"

@st.cache_data
def load_dataset():
    dataset = []
    try:
        with open(TEST_CSV_FILE, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    try:
                        dataset.append((int(row[0]), row[1].strip()))
                    except ValueError:
                        continue
    except FileNotFoundError:
        st.error(f"{TEST_CSV_FILE} not found!")
    return dataset


# ==============================
# Helper: Render dynamic parameter widgets
# ==============================
def render_param_widget(param_name, param_info, current_value):
    param_type = param_info["type"]

    if param_type == "int":
        return st.number_input(
            label=param_name,
            min_value=param_info.get("min", 0),
            max_value=param_info.get("max", 100),
            value=current_value,
            step=param_info.get("step", 1)
        )
    elif param_type == "float":
        return st.number_input(
            label=param_name,
            min_value=param_info.get("min", 0.0),
            max_value=param_info.get("max", 100.0),
            value=current_value,
            step=param_info.get("step", 0.1),
            format="%.4f"
        )
    elif param_type == "select":
        return st.selectbox(label=param_name, options=param_info.get("options", []), index=param_info.get("options", []).index(current_value))
    elif param_type == "file":
        return st.text_input(label=param_name, value=current_value)
    elif param_type == "list":
        # simple comma-separated input
        text = st.text_input(label=param_name, value=",".join(map(str, current_value)))
        return [int(x.strip()) if x.strip().isdigit() else x.strip() for x in text.split(",")]
    else:
        st.write(f"Unsupported param type: {param_type}")
        return current_value


# ==============================
# Main Streamlit App
# ==============================
st.title("🧩 Algorithm Explorer")
dataset = load_dataset()
if not dataset:
    st.stop()

algo_names = manager.get_available_algos()
selected_algo_name = st.sidebar.selectbox("Select Algorithm", algo_names)
manager.select(selected_algo_name)
algo = manager.get_current()

st.header(f"Algorithm: {selected_algo_name}")
st.write(algo.description)

# Display and update parameters dynamically
st.subheader("Parameters")
new_params = {}
for param_name, param_info in algo.param_schema.items():
    current_value = algo.params.get(param_name, param_info.get("default"))
    new_value = render_param_widget(param_name, param_info, current_value)
    new_params[param_name] = new_value

if st.button("✅ Fit Algorithm"):
    algo.set_params(**new_params)
    with st.spinner("Fitting..."):
        algo.fit(dataset)
    st.success("Algorithm fitted successfully!")

# Allow user to test predictions
st.subheader("Test Predictions")
sample_text = st.text_area("Enter tweet/text", "This is an example tweet.")
if st.button("Predict"):
    try:
        prediction = algo.predict_one(sample_text)
        st.write(f"Prediction: **{prediction}**")
    except Exception as e:
        st.error(f"Prediction failed: {type(e).__name__} - {e}")
