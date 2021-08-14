import streamlit as st
import pandas as pd
from random import randint
import base64

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    return f'<a href="data:file/csv;base64,{b64}" download="answers.csv">Download answers.csv</a>'

st.title('Host classifier by V3.0')
with st.form('text'):
	text_input = st.text_area('Enter host name here: ', 'yourHost.com')
	submit_button = st.form_submit_button('Predict is that a technical host')
	if submit_button:
		predict = randint(0,1)
		st.write('Predicted label: ', f'{predict}')

st.write('OR')
with st.form('file'):
	uploaded_file = st.file_uploader("Choose a csv file", ["csv"])
	file_button = st.form_submit_button('Predict labels')
	if uploaded_file:
		file = pd.read_csv(uploaded_file)
		if file_button:
				ans=file.describe()
				st.write(ans.head())
				st.markdown(get_table_download_link(ans), unsafe_allow_html=True)


				