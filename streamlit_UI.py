import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64


#To validate the Input File Template
def validate_csv_columns(df):
    # Define the expected columns for validation
    expected_columns = ["Type", "Air temperature", "Process temperature", "Rotational speed", "Torque", "Tool wear"]    
    # Check if all expected columns are present in the DataFrame
    missing_columns = set(expected_columns) - set(df.columns)
    if missing_columns:
        st.error(f"The uploaded CSV is missing the following columns: {', '.join(missing_columns)}")
        return False

    return True

# To display the DataFrame of successful samples on click
def display_successful_samples(df):
    st.write(df[df['Prediction'] == 0])


#To display file upload data in right
def process_uploaded_file(file):
    if file is not None:
        if file.type == 'application/vnd.ms-excel':
            st.warning("Please upload a CSV file.")
        else:
            # Read the uploaded CSV file
            df = pd.read_csv(file)
            # Validate the columns of the uploaded CSV
            if validate_csv_columns(df):
                # Display the content in the right column
                st.success("CSV file uploaded and validated.")
                with right_col:
                    # st.header('Machine Metrics')
                    st.markdown("<h1 style='font-size: 20px;'>Machine Metrics</h1>", unsafe_allow_html=True)
                    # st.write(df)
                    type_dict = {'L': 0, 'M': 1, 'H': 2}
                    cause_dict = {
                        0: 'No Failure',
                        1: 'Power Failure',
                        2: 'Overstrain Failure',
                        3: 'Heat Dissipation Failure',
                        4: 'Tool Wear Failure'
                    }
                    features = ["Type", "Air temperature", "Process temperature", "Rotational speed", "Torque", "Tool wear"]
                    df["Type"] = df["Type"].apply(lambda x: type_dict.get(x, -1))

                    with open('predictive_maintenance_model_failure.pkl', 'rb') as file:
                        loaded_model = pickle.load(file)
                        proba_predictions = loaded_model.predict_proba(df)
                        prediction = loaded_model.predict(df)
                        df['Prediction'] = prediction
                        df['Probability'] = proba_predictions[:, 1] * 100
                        
                    print('df.columns',df.columns)
                    df.loc[:9, 'Prediction'] = 0
                    if 'Prediction' in df.columns:
                        with open('predictive_maintenance_model_failure_type.pkl', 'rb') as file:
                            loaded_failure_type_model = pickle.load(file)
                        loaded_model = loaded_failure_type_model['model']
                        loaded_scaler = loaded_failure_type_model['scaler']
                        num_features = [feature for feature in features if feature not in ['Type', 'Prediction','Probability']]
                        # st.write(num_features)
                        df[num_features] = loaded_scaler.transform(df[num_features])
                        exclu_pred = [feature for feature in features if feature not in ['Prediction','Probability']]
                        # Generate failure_type using the loaded model
                        failure_type = loaded_model.predict(df[exclu_pred])
                        # st.write(failure_type)
                        df['Failure Type'] = [cause_dict.get(val, 'Unknown') for val in failure_type]
                    else:
                        st.write("No Failure Type data available")

                    total_samples = len(df)
                    success_count = (df['Prediction'] == 0).sum()
                    failure_count = (df['Prediction'] == 1).sum()
                    success_percentage = (success_count / total_samples) * 100
                    failure_percentage = (failure_count / total_samples) * 100

                    # st.write("Total Samples:", total_samples)
                    Total_Samples_button = st.button(f"Total Samples: {total_samples}",disabled=True)                    
                    # Display "Success Count" as a hyperlink
                    # success_count_link = f"<a href='#' class='css-164nlkn'>{success_count}</a>"
                    # st.write("Success Count:",success_count_link, unsafe_allow_html=True)
                    # Toggle the display of successful samples when the link is clicked
                    # display_successful_samples(df[df['Prediction'] == 0])

                    # Use an expander to toggle the display of successful samples
                    session_state = st.session_state
                    # Initialize show_expander flag if it doesn't exist
                    if 'show_expander' not in session_state:
                        session_state.show_expander = False

                    success_count_button = st.button(f"Success Count: {success_count}")
                    if success_count_button:
                        session_state.show_expander = not session_state.show_expander
                    if session_state.show_expander:
                        display_successful_samples(df[df['Prediction'] == 0])

                   
                    Failure_Count_button = st.button(f"Failure Count: {failure_count}",disabled=True)
                    SuccessPercentage_button = st.button(f"Success Percentage: {success_percentage} %",disabled=True)
                    Failure_Percentage_button = st.button(f"Failure Percentage: {failure_percentage} %",disabled=True)
                    # st.write("Failure Count:", failure_count)
                    # st.write("Success Percentage:", success_percentage, "%")
                    # st.write("Failure Percentage:", failure_percentage, "%")

                    # Inside the right_col context manager
                    if 'Failure Type' in df.columns:

                        # Get unique failure types
                        unique_failure_types = df['Failure Type'].unique()

                        # st.header("Failure Types")
                        st.markdown("<h1 style='font-size: 20px;'>Failure Types</h1>", unsafe_allow_html=True)

                        # Display each failure type as a link
                        for failure_type in unique_failure_types:
                            selected_data = df[df['Failure Type'] == failure_type]
                            # len_style = f"color: {'red' if len(selected_data) > 100 else 'orange' if len(selected_data) > 50 else 'black'};"
                            link_text = f"{failure_type}: **({len(selected_data)})**"
                            if st.button(link_text):
                                # st.header(f"Data for {link_text}")
                                st.write(selected_data)
                            
                            

                        # Group the data by 'Failure Type' and 'Prediction' and count the occurrences
                        grouped_data = df.groupby(['Failure Type', 'Prediction']).size().reset_index(name='Count')

                        # Create a pie chart using go.Figure
                        fig = go.Figure(data=[go.Pie(labels=grouped_data['Failure Type'], values=grouped_data['Count'])])

                        st.header("Failure Distribution")

                        # Add customdata to each trace
                        fig.update_traces(customdata=grouped_data['Failure Type'])

                        # Create a Plotly event handler to display data when a pie slice is clicked
                        def update_on_click(trace, points, selector):
                            if points.point_inds:
                                selected_failure_type = points.point.customdata
                                selected_data = df[df['Failure Type'] == selected_failure_type]
                                st.header(f"Data for Failure Type: {selected_failure_type}")
                                st.write(selected_data)

                        # Add an event handler to the layout for click events
                        fig.update_layout(clickmode='event+select')
                        fig.for_each_trace(lambda trace: trace.on_click(update_on_click))

                        st.plotly_chart(fig, use_container_width=True)
                        # # Create a dropdown menu to select the "Failure Type"
                        # selected_failure_type = st.selectbox("Select Failure Type:", grouped_data['Failure Type'])

                        # # Define a callback function for displaying data when a type is selected
                        # def display_data(selected_failure_type):
                        #     selected_data = df[df['Failure Type'] == selected_failure_type]
                        #     st.header(f"Data for Failure Type: {selected_failure_type}")
                        #     st.write(selected_data)

                        # # Display data when a "Failure Type" is selected from the dropdown
                        # display_data(selected_failure_type)


# Define a function to display the input fields
def display_input_fields():
    logo_col, title_col = st.columns((1, 2))
    with logo_col:
        st.markdown("<h1 style='font-size: 10px; margin-top:-30px;'>@Powered By</h1>", unsafe_allow_html=True)
        image_path = os.path.join('images', f'logo.jpg')
        if not os.path.exists(image_path):
            st.warning("Image not found.")
        else:
            # Display the image
            st.image(image_path, width=60)
    with title_col:
        # st.title("Predictive Maintenance", className="predictive-title")
        st.markdown("<h1 style='font-size: 30px;margin-left:-70px;'>Predictive Maintenance</h1>", unsafe_allow_html=True)
    with st.expander("Bulk Machine check", expanded=False):
        # file_upload_button = st.file_uploader("Upload a file")
        file_upload_button = st.file_uploader("Upload a CSV file", type=["csv"])
        if file_upload_button is not None:
            process_uploaded_file(file_upload_button)

    with st.expander("Individual Machine check", expanded=False):
        # st.header("Machine Parameters")
        st.markdown("<h1 style='font-size: 20px;'>Machine Parameters</h1>", unsafe_allow_html=True)
        machine_type = st.selectbox("Type (L, M, or H):", ['L', 'M', 'H'])
        air_temp = st.number_input("Air Temperature:", step=0.1)
        process_temp = st.number_input("Process temperature:", step=0.1)
        rotational_speed = st.number_input("Rotational speed:", step=0.1)
        torque = st.number_input("Torque:", step=0.1)
        tool_wear = st.number_input("Tool wear:", step=0.1)
        submit_button = st.button("Submit")

    return machine_type, air_temp, process_temp, rotational_speed, torque, tool_wear, submit_button

# Define a function to load and run the predictive model
def run_predictive_model(machine_type, air_temp, process_temp, rotational_speed, torque, tool_wear):
    with right_col:
        # st.title("Machine Failure Information")
        st.markdown("<h1 style='font-size: 20px;'>Machine Failure Information</h1>", unsafe_allow_html=True)
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
            st.error("Type should be 'L', 'M', or 'H.")
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

        features = ["Type", "Air temperature", "Process temperature", "Rotational speed", "Torque", "Tool wear"]

        with st.expander("Input Data", expanded=True):
            st.write(df)    

        with open('predictive_maintenance_model_failure.pkl', 'rb') as file:
            loaded_model = pickle.load(file)
            prediction = loaded_model.predict(df)
            

        with st.expander("Machine Failure Prediction", expanded=True):
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
                with open('predictive_maintenance_model_failure_type.pkl', 'rb') as file:
                    loaded_failure_type_model = pickle.load(file)
                loaded_model = loaded_failure_type_model['model']
                loaded_scaler = loaded_failure_type_model['scaler']

                num_features = [feature for feature in features if df[feature].dtype == 'float64']
                df[num_features] = loaded_scaler.transform(df[num_features])

                failure_type = loaded_model.predict(df)

                failure_type_str = str(failure_type).strip('[]')
                failure_type = cause_dict.get(failure_type_str, 'Unknown')

                # with st.expander("Failure Type and Image", expanded=True):
                st.write("Failure Type:")
                st.write(failure_type)

                # Check if the image file exists in the 'images' folder
                image_path = os.path.join('images', f'{failure_type}.jpg')
                if not os.path.exists(image_path):
                    st.warning("Image not found.")
                else:
                    # Display the image
                    st.image(image_path, caption=failure_type, width=150)

# Main application code
st.set_page_config(
    page_title="Machine Information",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create a layout with two columns using st.beta_columns
left_col, right_col = st.columns((4,6))

# Apply background color and CSS styles to the left column
with left_col:
    st.markdown(
        """
        <style>
        .st-emotion-cache-1yycg8b{
            background-color: #B4C4D2;
            padding: 20px;
            height:600px;
            max-height: 90vh; /* Max height of the viewport */
            overflow-y: auto; /* Scroll if content overflows */
        }
        .st-emotion-cache-ocqkz7 {
            gap:0em;
            padding-bottom:0px;
            margin-bottom:0px;
        }
        .block-container.st-emotion-cache-z5fcl4{
            margin-top:0px;
            padding:2rem;
            padding-bottom:0px;
            margin-bottom:0px;
        }
        .css-164nlkn{
            padding-top:0px;
            margin-top:0px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    machine_type, air_temp, process_temp, rotational_speed, torque, tool_wear, submit_button = display_input_fields()

# Apply background color and CSS styles to the right column
with right_col:
    st.markdown(
        """
        <style>
        .st-emotion-cache-fplge5 {
            background-color: lightgray;
            padding: 20px;
            height:600px;
            max-height: 90vh; /* Max height of the viewport */
            overflow-y: auto; /* Scroll if content overflows */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    

if submit_button:
    run_predictive_model(machine_type, air_temp, process_temp, rotational_speed, torque, tool_wear)
