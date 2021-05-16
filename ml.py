import streamlit as st 
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
#from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import model_selection
#from sklearn.preprocessing import LabelEncoder
matplotlib.use('Agg')

from PIL import Image

#Set title

st.title('Automatic Data Analysis')
image = Image.open('msruas.png')
st.image(image,use_column_width=True)



def main():
	activities=['EDA','Visualisation','model','About us']
	option=st.sidebar.selectbox('Selection option:',activities)

	


#DEALING WITH THE EDA PART


	if option=='EDA':
		st.subheader("Exploratory Data Analysis")

		data=st.file_uploader("Upload dataset:",type=['csv','xlsx','txt','json'])
		
		if data is not None:
			st.success("Data successfully loaded")
			df=pd.read_csv(data)
			st.dataframe(df.head(50))

			if st.checkbox("Display shape"):
				st.write(df.shape)
			if st.checkbox("Display columns"):
				st.write(df.columns)
			# if st.checkbox("Select multiple columns"):
			# 	selected_columns=st.multiselect('Select preferred columns:',df.columns)
			# 	df1=df[selected_columns]
			# 	st.dataframe(df1)

			if st.checkbox("Display summary"):
				st.write(df.describe().T)

			if st.checkbox('Display Null Values'):
				st.write(df.isnull().sum())

			if st.checkbox("Display the data types"):
				st.write(df.dtypes)
			if st.checkbox('Display Correlation of data variuos columns'):
				st.write(df.corr())




#DEALING WITH THE VISUALISATION PART


	elif option=='Visualisation':
		st.subheader("Data Visualisation")

		data=st.file_uploader("Upload dataset:",type=['csv','xlsx','txt','json'])
		st.success("Data successfully loaded")
		if data is not None:
			df=pd.read_csv(data)
			st.dataframe(df.head(50))

			if st.checkbox('Select Multiple columns to plot'):

				selected_columns=st.multiselect('Select your preferred columns',df.columns)
				df1=df[selected_columns]
				st.dataframe(df1)

			seed=st.sidebar.slider('Start Value',0,df.shape[0])
			seed2=st.sidebar.slider('End Value',1,df.shape[0],value=100)

			columns = df.columns.tolist()
			
			if st.checkbox('Display Heatmap'):
				fig, ax = plt.subplots(figsize=(15,5))
				st.write(sns.heatmap(df1[seed:seed2].corr(),vmax=1,square=True,annot=True,cmap='viridis'))
				st.pyplot(fig)
			if st.checkbox('Display Pairplot'):
				st.write(sns.pairplot(df1[seed:seed2],diag_kind='kde'))
				st.set_option('deprecation.showPyplotGlobalUse', False)
				st.pyplot()
			if st.checkbox('Box Plot'):
				st.write(sns.boxplot(palette="pastel", data=df1))
				st.pyplot()
			if st.checkbox('Bar Chart'):
				st.bar_chart(df1[seed:seed2])
				st.set_option('deprecation.showPyplotGlobalUse', False)
				
			if st.checkbox('Line Chart'):
				st.line_chart(df1[seed:seed2])
				st.set_option('deprecation.showPyplotGlobalUse', False)
				
			if st.checkbox('Area Chart'):
				st.area_chart(df1[seed:seed2])
				st.set_option('deprecation.showPyplotGlobalUse', False)
			
				            
				






	# DEALING WITH THE MODEL BUILDING PART

	








#DELING WITH THE ABOUT US PAGE



	elif option=='About us':

		# st.markdown('This is an interactive web page for our ML project, feel feel free to use it. This dataset is fetched from the UCI Machine learning repository. The analysis in here is to demonstrate how we can present our wok to our stakeholders in an interractive way by building a web app for our machine learning algorithms using different dataset.'
		# 	)


		st.balloons()
	# 	..............


if __name__ == '__main__':

	main() 



