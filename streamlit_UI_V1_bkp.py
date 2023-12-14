import streamlit as st
import pandas as pd
import pickle
import os
from PIL import Image

# Define a function to display the input fields
def display_input_fields():
    st.title("Machine Failure Information")
    st.sidebar.header("Machine Parameters")

    # Type selection
    machine_type = st.sidebar.selectbox("Type (L, M, or H):", ['L', 'M', 'H'])

    # Other input fields
    air_temp = st.sidebar.number_input("Air Temperature:", step=0.1)
    process_temp = st.sidebar.number_input("Process temperature:", step=0.1)
    rotational_speed = st.sidebar.number_input("Rotational speed:", step=0.1)
    torque = st.sidebar.number_input("Torque:", step=0.1)
    tool_wear = st.sidebar.number_input("Tool wear:", step=0.1)

    submit_button = st.sidebar.button("Submit")

    return machine_type, air_temp, process_temp, rotational_speed, torque, tool_wear, submit_button

# Define a function to load and run the predictive model
def run_predictive_model(machine_type, air_temp, process_temp, rotational_speed, torque, tool_wear):
    type_dict = {'L': 0, 'M': 1, 'H': 2}
    cause_dict = {
        '0': 'No Failure',
        '1': 'Power Failure',
        '2': 'Overstrain Failure',
        '3': 'Heat Dissipation Failure',
        '4': 'Tool Wear Failure'
    }
    
    machine_type = machine_type.upper()
    
    if machine_type not in ['L', 'M', 'H']:
        st.error("Type should be 'L', 'M', or 'H'.")
        return

    data = {
        "Type": [type_dict[machine_type]],
        "Air temperature": [air_temp],
        "Process temperature": [process_temp],
        "Rotational speed": [rotational_speed],
        "Torque": [torque],
        "Tool wear": [tool_wear]
    }
    df = pd.DataFrame(data)

    features = ["Type","Air temperature", "Process temperature", "Rotational speed", "Torque", "Tool wear"]
    st.write("Input Data:")
    st.write(df)

    with open('predictive_maintenance_model_failure.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
        prediction = loaded_model.predict(df)

    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; border: 2px solid black; padding: 10px;">
            <div style="color: blue; font-size: 18px;">Machine Failure:</div>
            <div style="color: {'red' if prediction == 1 else 'green'}; font-size: 18px;">{'Failed' if prediction == 1 else 'No Failure'}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if prediction == 1:
        # with open('predictive_maintenance_model_failure_type.pkl', 'rb') as file:
        #     loaded_failure_type_model = pickle.load(file)
        #      = loaded_failure_type_model.predict(df)

        with open('predictive_maintenance_model_failure_type.pkl', 'rb') as file: 
            loaded_failure_type_model = pickle.load(file)
        loaded_model = loaded_failure_type_model['model']
        loaded_scaler = loaded_failure_type_model['scaler']

        num_features =[feature for feature in features if df[feature].dtype=='float64']
        df[num_features]=loaded_scaler.transform(df[num_features])

        failure_type=loaded_model.predict(df)

        failure_type_str = str(failure_type).strip('[]')
        failure_type = cause_dict.get(failure_type_str, 'Unknown')
        # Initialize the image_html variable
        image_html = ""

        # Check if the image file exists in the 'images' folder
        image_path = os.path.join('images', f'{failure_type}.jpg')
        if not os.path.exists(image_path):
            st.warning("Image not found.")
        else:
            # Construct an HTML string to display the image and text
            # image_html = f'<img src="{image_path}" style="max-width:100%;"><br>{failure_type}'
            # Display the image
            st.image(image_path,caption=failure_type)
            st.write(
                f'<style>img {{width: {150}px; height: {150}px;margin-top: {20}px;}}</style>',
                unsafe_allow_html=True
            )

        # st.markdown(
        #     f"""
        #     <div style="display: flex; justify-content: space-between; align-items: center; border: 2px solid black; padding: 10px;">
        #         <div style="color: blue; font-size: 18px;">Type of Failure:</div>
        #         <div style="color: red;font-size: 18px;">{image_html}</div>
        #     </div>
        #     """,
        #     unsafe_allow_html=True,
        # )

# Main application code
st.set_page_config(
    page_title="Machine Information",
    layout="wide",
    initial_sidebar_state="expanded"
)

    

# Custom CSS to adjust the margin-top for sections
st.markdown(
    """
    <style>
    .css-1544g2n.e1fqkh3o4,.block-container.css-z5fcl4.egzxvld4{
        padding-top:30px;
    }
    .main.css-uf99v8.egzxvld5{
        background-color: #B4C4D2;
    }
    .css-18ni7ap.e8zbici2{
        background-color: #B4C4D2;
    }
    .css-11gayxo{
        gap:0.6rem;
    }
    .css-81oif8.effi0qh3{
        margin-bottom:0px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

machine_type, air_temp, process_temp, rotational_speed, torque, tool_wear, submit_button = display_input_fields()

if submit_button:
    run_predictive_model(machine_type, air_temp, process_temp, rotational_speed, torque, tool_wear)