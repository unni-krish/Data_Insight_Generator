import streamlit as st
import os
import base64
import markdown2
from xhtml2pdf import pisa
from io import BytesIO
import json
import insight_llm  


with open('./json/prompt.json', 'r') as f:
    prompts = json.load(f)


def generate_insights(df):
    insights = {}
    
    # Null value analysis
    insights['null_values'] = df.isnull().sum()
    insights['null_percentage'] = (df.isnull().sum() / len(df)) * 100
    
    # Correlation analysis
    insights['correlation_matrix'] = df.corr()
    
    # Summary statistics
    insights['summary_stats'] = df.describe()
    
    # Unique value counts
    insights['unique_values'] = df.select_dtypes(include=['object']).nunique()
    
    # Outlier detection
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    insights['outliers'] = df[(df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))]
    
    # Skewness and Kurtosis
    insights['skewness'] = df.skew()
    insights['kurtosis'] = df.kurtosis()
    
    # Distribution analysis
    distribution = {}
    for column in df.select_dtypes(include=['float', 'int']).columns:
        distribution[column] = df[column].value_counts(bins=10)
    insights['distribution'] = distribution
    
    # Missing data pattern
    insights['missing_data_pattern'] = df.isnull().mean(axis=1).value_counts()
    
    # Correlation with target variable (if specified)
    if 'target_column' in df.columns:
        insights['target_correlation'] = df.corr()['target_column'].sort_values(ascending=False)
    
    return insights





def convert_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')
    
def convert_html(df, result, path):
    insights = generate_insights(df)
    plot_imgs = os.listdir(path)
    table_html = df.head().to_html(index=False, classes='table table-striped', border=1)

    des_pmt = prompts['des_pmt'].format(df.head().to_string(index=False))
    description = insight_llm.llm(
        prompts['des_sys_pmt'],
        des_pmt
    )

    con_sys = prompts['con_pmt'].format(
    null_values=insights['null_values'].to_dict(),
    null_percentage=insights['null_percentage'].to_dict(),
    correlation_matrix=insights['correlation_matrix'].to_dict(),
    summary_stats=insights['summary_stats'].to_dict(),
    unique_values=insights['unique_values'].to_dict(),
    outliers=insights['outliers'].dropna().to_dict(),
    skewness=insights['skewness'].to_dict(),
    kurtosis=insights['kurtosis'].to_dict(),
    distribution=insights['distribution'],
    missing_data_pattern=insights['missing_data_pattern'].to_dict()
)
    conclusion = insight_llm.llm(prompts['con_sys_pmt'],con_sys)
    conclusion_html = markdown2.markdown(conclusion)
    description_html =  markdown2.markdown(description)
    # Generate the HTML content with the dataset table on top

    
    html_content = f"""

    <style>
    .page-break {{
        page-break-before: always;
    }}

    .table.table-striped td, .table.table-striped th {{
        text-align: center;
        height: 50px;
    }}
    </style>

    <div style='background-color:white;'>
        <h1 style='background-color:purple;color:white;padding:10px;text-align:center;'>Dataset Overview</h1>
        <div style='overflow-x:auto; width: 100%; '>
            {table_html}
            <br>
            <br>
            <p style ='text-align:left;padding-left:5px;padding-right:5px;font-size: larger; font-weight: bold;'>{description_html}</p>
        </div>
        <div class = "page-break">
            <h3 style='background-color:purple;color:white;padding:10px;text-align:center;'>List of Plots</h3>
            <h4 style='text-align:center;'> <u><b>These are the plots utilized in this report</b></u> </h4>
            {"".join([
                f"<h4 style='text-align:left;padding-left:5px;'>{index + 1}. {plot['plot_name']}</h4>"
                for index, plot in enumerate(result['plots'])
            ])}
        </div>
        {"".join([
            f"<div class='page-break'>"
            f"<h3 style='background-color:purple;color:white;padding:10px;text-align:center;'>{plot_img.split('.png')[0]}</h3>"
            f"<img src='data:image/png;base64,{convert_image_to_base64(os.path.join('static/plots', plot_img))}' alt='Plot {i + 1}' style='width:100%;'>"
            f"<br>"
            f"<br>"
            f"<p style ='text-align:left;padding-left:5px;padding-right:5px;font-size: larger; font-weight: bold;'>{[plot['reason'] for plot in result['plots'] if plot['plot_name'] == plot_img.split('.png')[0]][0]}</p>"
            f"</div>"
            for i, plot_img in enumerate(plot_imgs)
        ])}
        <div class = "page-break">
         <h3 style='background-color:purple;color:white;padding:10px;text-align:center;'>Conclusion</h3>
         <p style ='text-align:left;padding-left:5px;padding-right:5px;font-size: larger; font-weight: bold;'>{conclusion_html}</p>
        </div>
    </div>


    """

    # Render the HTML content in Streamlit
    st.components.v1.html(html_content, height=800,width=800, scrolling=True)

        # Convert HTML to PDF
    pdf_buffer = BytesIO()
    pisa_status = pisa.CreatePDF(BytesIO(html_content.encode('utf-8')), dest=pdf_buffer)
    pdf_buffer.seek(0)

    # Provide a download button for the PDF
    col1,col2,col3 = st.columns(3)
    with col2:
        st.download_button(
            label="Download report as PDF",
            data=pdf_buffer,
            file_name="report.pdf",
            mime="application/pdf"
        )
