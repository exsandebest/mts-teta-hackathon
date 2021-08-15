import streamlit as st
import pandas as pd
from random import randint
import base64
import joblib


def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    return f'<a href="data:file/csv;base64,{b64}" download="answers.csv">Download answers.csv</a>'

model = joblib.load('models/main_model.pickle')
target_name = ['personal','technical']

def data_prep(t):
	df = t.copy()
	df["start_with_api"] = df["url"].str.contains("^api", regex=True).astype(int)
	df["has_userapi"] = df["url"].str.contains("userapi").astype(int)
	df["has_googleapis"] = df["url"].str.contains("googleapis").astype(int)
	df["size_of_url"] = df["url"].apply(lambda x: len(x))
	df["size_of_url_split"] = df["url"].apply(lambda x: len(x.split(".")))
	df["clear_url"] =  df["url"].apply(lambda x: x.replace(".", " "))
	df["minus_count"] = df["url"].str.count("-")
	return df



st.title('Demo of host classifier by _V3.0_')
with st.form('text'):
	text_input = st.text_area('Enter host name here: ', 'yourHost.com')
	submit_button = st.form_submit_button('Predict label')
	if submit_button:
		if ' ' not in text_input and '.' in text_input and text_input[0]!='.':
			raw_data = pd.DataFrame({'url':[text_input]})
			test_data = data_prep(raw_data)
			predict = model.predict(test_data.drop(columns=["url"]))
			st.markdown(f'Predicted label: **{target_name[predict[0]]}**' 
				' (probability: {:.3f} )'.format(model.predict_proba(test_data.drop(columns=["url"]))[0][predict][0]))
		else:
			st.write('Enter **correct** host')


st.write('OR')


with st.form('file'):
	uploaded_file = st.file_uploader("Upload a csv file", ["csv"])
	file_button = st.form_submit_button('Predict labels')
	if uploaded_file:
		file = pd.read_csv(uploaded_file)
		if file_button:
			clmn = file.columns
			if len(clmn)!=1 and 'url' not in clmn:
				st.write('Wrong format of data, please upload data with column named "url" or with only one text column')
			else:
				if len(clmn)==1:
					file.columns=['url']
				try:
					prep_file = pd.DataFrame({'url':file['url'].values})
					test_data = data_prep(prep_file)
					predict = model.predict(test_data.drop(columns=["url"]))
					prep_file['Prediction']=predict
					st.write(prep_file)
					st.markdown(get_table_download_link(prep_file), unsafe_allow_html=True)
				except:
					st.write('Oops! You data format is wrong!\nPlease make sure your data constis of strings!')



				