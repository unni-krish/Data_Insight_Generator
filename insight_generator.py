import streamlit as st
import pandas as pd
import os
import shutil
import json
import sys

sys.path.append('./Module')
import LmPlot
import html_redering as hr
import insight_llm  

# Function to load the uploaded file
def load_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        data = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file type")
        return None
    return data

# Function to generate response from LLM
def generate_llm_response(prompt):
    response = insight_llm.llm(prompt)  # Assuming you have a function to get response from LLM
    return response



def prompts(data,target_column):
    
    with open('json/plots_category.json', 'r') as file:
        plot_category = json.load(file)

    with open('json/prompt.json', 'r') as file:
        prompts_json = json.load(file)
    
    
    if data[target_column].nunique() == 2:
        plot_names = plot_category['binary']
    elif (data[target_column].nunique() > 2) and (data[target_column].nunique() <= 10):
        plot_names = plot_category['multi']
    elif data[target_column].nunique() >10:
        plot_names = plot_category['regressor']

    system_prompt = prompts_json["plot_exp_sys"]

    prompt = prompts_json["plot_exp_pmt"].format(data=data,target_column=target_column,plot_names=plot_names)

    return system_prompt,prompt



def clear_directory(directory_path):

    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.unlink(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

st.markdown(
    """
    <style>
    .title {
        text-align: center;
        border: 2px solid #a7189c;
        border-radius: 15px;
        padding: 10px;
        color: #a7189c;

    }
    .stButton button {
        width: 100%;
        display: block;
        margin: 0 auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Streamlit app
def main():
    st.markdown('<h1 class="title">Data Insight Generator</h1><br>', unsafe_allow_html=True)
    

    
    uploaded_file = st.file_uploader("Upload a CSV or Excel files", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if data is not None:
            st.write("File loaded successfully.")
            st.write(data)

            if 'plots' not in st.session_state:
                st.session_state.plots = []  # Store plots in session state

            # Interface for selecting columns and plot type
            st.sidebar.header("Plot Configuration")
            data_model = st.sidebar.selectbox("Select Dataset Type", ['Classification','Regression'])
            # model = st.sidebar.selectbox("Select a Machine learning Model", data_models[data_model])
            target_column = st.sidebar.selectbox("Select Target Column", data.columns)
            with st.sidebar:
                toggle = False #st.toggle('Load Json')
            # Button to create plot
            # col_1,col_2,col_3 = st.sidebar.columns(3)
            if st.sidebar.button("Generate Report"):
                x = data.drop(target_column,axis=1)
                y = data[target_column]

                
                if toggle:
                    with open('plots.json', 'r') as file:
                        json_result = json.load(file)
                        st.toast('Json Result Generated Successfully', icon="✅")
                else:
                    system_prompt,prompt = prompts(data.head(),target_column)
                    result = insight_llm.llm(system_prompt,prompt)

                    st.toast('Json Result Generated Successfully', icon="✅")
                    json_result = eval(result.split('```json')[1].split('```')[0])

                    print('json_result : ',json_result)
                    print('json_result : ',len(json_result['plots']))


                # Load the JSON file
                with st.spinner():
                    plots = json_result['plots']
                    for plot in plots:
                        plot_name = plot['plot_name']
                        LmPlot.Generate_Plots(data,x,y,plot_name,data_model,target_column)
                    path = 'static/plots'
                    hr.convert_html(data,json_result,path)
                    st.toast('Report Generated Successfully', icon="✅")
                    clear_directory(path)
                # print(result)
                
               
if __name__ == "__main__":
    main()
