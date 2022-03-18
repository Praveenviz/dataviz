import streamlit as st 
import numpy as np 
import pandas as pd
import seaborn as sns
import altair as alt
from fpdf import FPDF
import base64
from tempfile import NamedTemporaryFile
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics
#from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import model_selection
import plotly.graph_objs as go
import plotly.express as px
#from sklearn.preprocessing import LabelEncoder
matplotlib.use('Agg')

from PIL import Image

#Set title

st.title('Automatic Data Analysis')
image = Image.open('lpu.png').convert('RGB').save('new.jpeg')
st.image(image,use_column_width=True)

def create_download_link(val, filename):
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'

def main():
	activities=['EDA','Visualisation','model(classifier)','model(Regressor)','About us']
	option=st.sidebar.selectbox('Selection option:',activities)

	def Convert(string):
	    li = list(string.split(","))
	    return li






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
		
		if data is not None:
			st.success("Data successfully loaded")
			df=pd.read_csv(data)
			st.dataframe(df.head(50))
			


			if st.checkbox('Select Multiple columns to plot'):



				selected_columns=st.multiselect('Select your preferred columns',df.columns)
				df1=df[selected_columns]
				st.dataframe(df1)

				coll = {
				'Category': [i for i in df1.columns],
        		'Type': [j for j in df1.dtypes]
				}
				df2 = pd.DataFrame(coll, columns = ['Category', 'Type'])
				col_lis = []
				
				int_lis = []
				for i in df2.index:
					if df2['Type'][i] == object:
					 	col_lis.append(df2['Category'][i])
					elif df2['Type'][i] == 'int64':
					 	int_lis.append(df2['Category'][i])
					elif df2['Type'][i] == 'float64':
						int_lis.append(df2['Category'][i])
					else:
						continue;
					
		

				name=st.sidebar.selectbox('Select your preferred classifier:',col_lis)

				
				seed=st.sidebar.slider('Start Value',0,df.shape[0])
				seed2=st.sidebar.slider('End Value',1,df.shape[0],value=100)

				if name:
					tot_lis = Convert(name) + int_lis
					customers = df1[seed:seed2][tot_lis]
					customer_group = customers.groupby(name)
					sales_totals = customer_group.sum()
						

			

			columns = df.columns.tolist()
			
			if st.checkbox('Display Heatmap'):
				fig, ax = plt.subplots(figsize=(15,5))
				st.write(sns.heatmap(df1[seed:seed2].corr(),vmax=1,square=True,annot=True,cmap='viridis'))
				st.pyplot(fig)
				int_lis[::]

			if st.checkbox('Display Pairplot'):
				st.write(sns.pairplot(df1[seed:seed2],diag_kind='kde'))
				st.set_option('deprecation.showPyplotGlobalUse', False)
				st.pyplot()

			if st.checkbox('Box Plot'):
				st.write(sns.boxplot(palette="pastel", data=df1))
				st.pyplot()

			if st.checkbox('Bar Chart'):
				if name:
					st.write(px.bar(sales_totals))

				else:
					
					st.bar_chart(df1[seed:seed2])
					st.set_option('deprecation.showPyplotGlobalUse', False)
				
			if st.checkbox('Line Chart'):
				if name:
					st.write(px.line(sales_totals))
				
				else:
					st.line_chart(df1[seed:seed2])
					st.set_option('deprecation.showPyplotGlobalUse', False)
				
			if st.checkbox('Area Chart'):
				if name:
					st.write(px.area(sales_totals))
				else:
					st.area_chart(df1[seed:seed2])
					st.set_option('deprecation.showPyplotGlobalUse', False)
			
				            
				



	# DEALING WITH THE MODEL BUILDING PART

	elif option=='model(classifier)':
		st.subheader("Model Building")

		data=st.file_uploader("Upload dataset:",type=['csv','xlsx','txt','json'])
		
		if data is not None:
			st.success("Data successfully loaded")
			df=pd.read_csv(data)
			st.dataframe(df.head(50))
			container = st.beta_container()
			all = st.checkbox("Select all")
			all_columns = []
			for i in df.columns:
				all_columns.append(i)

			try:
				if all:
				    selected_options = container.multiselect("Select one or more options:",all_columns,all_columns)
				    df1=df[selected_options]
				    st.dataframe(df1)
					
				else:
					new_data=container.multiselect("Select your preferred columns. NB: Let your target variable be the last column to be selected",df.columns)
					df1=df[new_data]
					st.dataframe(df1)
					

				
					#Dividing my data into X and y variables
					
				X=df1.iloc[:,0:-1]
				y=df1.iloc[:,-1]
			except:
				st.stop()

				
			seed=st.sidebar.slider('Seed',1,200)

			classifier_name=st.sidebar.selectbox('Select your preferred classifier:',('KNN','SVM','LR','naive_bayes','decision tree'))

			try:
				def add_parameter(name_of_clf):
					params=dict()
					if name_of_clf=='SVM':
						C=st.sidebar.slider('C',0.01, 15.0)
						params['C']=C
					elif name_of_clf=='KNN':
						K=st.sidebar.slider('K',1,20)
						params['K']=K
						leaf_size=st.sidebar.slider('leaf_size',1,10)
						params['leaf_size']=leaf_size
						P=st.sidebar.slider('P',1,20)
						params['P']=P
						return params

				#calling the function

				params=add_parameter(classifier_name)



				#defing a function for our classifier

				def get_classifier(name_of_clf,params):
					clf= None
					if name_of_clf=='SVM':
						clf=SVC(C=params['C'])
					elif name_of_clf=='KNN':
						clf=KNeighborsClassifier(n_neighbors=params['K'],leaf_size=params['leaf_size'],p=params['P'])		
					elif name_of_clf=='LR':
						clf=LogisticRegression()
					elif name_of_clf=='naive_bayes':
						clf=GaussianNB()
					elif name_of_clf=='decision tree':
						clf=DecisionTreeClassifier()
					else:
						st.warning('Select your choice of algorithm')

					return clf
				

				clf=get_classifier(classifier_name,params)


				X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=seed)
				
				clf.fit(X_train,y_train)
				

				y_pred=clf.predict(X_test)
				y_test1 = y_test.squeeze()
				pred = pd.DataFrame({'Actual' : y_test1.tolist(), 'Predicted' : y_pred.tolist()})
				st.write('Predictions:',pred)

				accuracy=accuracy_score(y_test,y_pred)

				st.write('Name of classifier:',classifier_name)
				st.write('Accuracy : ',accuracy*100)
				
				st.echo(classification_report(y_test,y_pred))


				if st.checkbox('Confusion Matrix'):
					y_col = y.unique()
					st.write('Confusion Matrix for:',classifier_name)
					cm = metrics.confusion_matrix(y_test,y_pred,labels=[y_col[0],y_col[1]])

					df_cm = pd.DataFrame(cm,index=[2,4],columns=['pred'+ str(y_col[0]),'pred' +str(y_col[1])])
					st.set_option('deprecation.showPyplotGlobalUse', False)
					plt.figure(figsize=(10,5))

					st.write(sns.heatmap(df_cm,annot=True))
					st.pyplot()
			except:
				st.info('Please select required columns in dataset to model the data')	
#Dealing with Reggression

	elif option=='model(Regressor)':
		st.subheader("Model Building")
		data=st.file_uploader("Upload dataset:",type=['csv','xlsx','txt','json'])

		if data is not None:
			st.success("Data successfully loaded")
			df=pd.read_csv(data)
			st.dataframe(df.head(50))
			container = st.beta_container()
			all = st.checkbox("Select all")
			all_columns = []
			for i in df.columns:
				all_columns.append(i)

			try:
				if all:
				    selected_options = container.multiselect("Select one or more options:",all_columns,all_columns)
				    df1=df[selected_options]
				    st.dataframe(df1)
					
				else:
					new_data=container.multiselect("Select your preferred columns. NB: Let your target variable be the last column to be selected",df.columns)
					df1=df[new_data]
					st.dataframe(df1)


					#Dividing my data into X and y variables

				X=df1.iloc[:,0:-1]
				y=df1.iloc[:,-1]
			except:
				st.stop()

			seed=st.sidebar.slider('Seed',1,200)

			classifier_name=st.sidebar.selectbox('Select your preferred regressor:',('KNN','SVM','LR','decision tree'))

			try:

				def add_parameter(name_of_clf):
					params=dict()
					if name_of_clf=='SVM':
						C=st.sidebar.slider('C',0.01, 15.0)
						params['C']=C
					elif name_of_clf=='KNN':
						K=st.sidebar.slider('K',1,15)
						params['K']=K
						return params

				#calling the function

				params=add_parameter(regressor_name)



				#defing a function for our classifier

				def get_regressor(name_of_clf,params):
					clf= None
					if name_of_clf=='SVM':
						clf=SVR()
					elif name_of_clf=='KNN':
						clf=KNeighborsRegressor(n_neighbors=params['K'])
					elif name_of_clf=='LR':
						clf=LinearRegression()
					elif name_of_clf=='decision tree':
						clf=DecisionTreeRegressor()
					else:
						st.warning('Select your choice of algorithm')

					return clf

				clf=get_regressor(regressor_name,params)

			
				X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=seed)

				clf.fit(X_train,y_train)


				y_pred=clf.predict(X_test)
				y_test1 = y_test.squeeze()
				pred = pd.DataFrame({'Actual' : y_test1.tolist(), 'Predicted' : y_pred.tolist()})
				st.write('Predictions:',pred)
				accuracy=clf.score(X_test,y_test)
				st.write('Name of regressor:',regr_name)
				st.write('Accuracy',accuracy)

			except:
				st.info('Please select required columns in dataset to model the data')



#DELING WITH THE ABOUT US PAGE



	elif option=='About us':

		# st.markdown('This is an interactive web page for our ML project, feel feel free to use it. This dataset is fetched from the UCI Machine learning repository. The analysis in here is to demonstrate how we can present our wok to our stakeholders in an interractive way by building a web app for our machine learning algorithms using different dataset.'
		# 	)


		st.balloons()
	# 	..............


if __name__ == '__main__':

	main() 



