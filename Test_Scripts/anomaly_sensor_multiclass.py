#%%[markdown]
## Linking Anomaly Detection With Multiclassification, A Juice Example.
# By: Christopher Smith
#
# When performing classification tasks with machine learning a common issue that can be faced with real-world data is "What if something that has never been seen before is introduced to the system?". What we would expect is that the model would try to predict it based on its data to the most similar class instead of identifying it as something new. <u> This is where anomaly detection comes in. What anomaly detection does is take a set of known data that we can call, values that we know exist and are true. </u> Then we can introduce a test set full of data values that it does not know, if the data looks very dismilar to the trained and known data then the model will call it an anomaly. Knowing this, we can create a 2-step machine learning approach whereby we train and anomaly detection system on known data that can then be subsetted into a mutliclassification task. Thereby, future occurences then will filter out unknown data measurements that are new to the system and not similar to the known and still allow us to predict on known cases. See below for an example of this using fruit data collected in the lab.
# 
# %%
# Import Libraries for modeling

# For Data processing
import numpy as np 
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.svm import OneClassSVM, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA

# For plotting
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns 
sns.set_style('ticks')

#%%[markdown]
#
# In this dataset are five classes of fruits, oranges, mandarins, tangerines, lemons, and limes. The data is prepared using 12 nanosensors, using a 1 hr endpoint for data collection, and is kept track of by a Date stamp and includes two blank columns as will be seen. The first steps done were loading the data and filtering out orange measurements prior to 2022. The reason is that the orange data extremely varies possibly due to a change in season or even supplier. Therefore, measurements of orange after 2022 are only considered.
#
# %%
# Import the data and visualize first 5 rows
df = pd.read_csv(r'C:\Users\chris\OneDrive\Visual Studio\Research\Data Analysis Citrus\Data Bank\anomaly_detection.csv')
df['Date'] = pd.to_datetime(df['Date']).dt.strftime("%Y-%m-%d")
df

# %%
# Drop orange samples before 2022
con1 = df['Label'] == 'orange'
con2 = df['Date']  <= '2022-01-01'

mask = con1 & con2

df = df.drop(df[mask].index)
df
# %%
# Remove Date, and unnamed columns, show first 5 rows to confirm
df = df.drop(['Date', 'Unnamed: 14', 'Unnamed: 15'], axis = 1)
df.head()

# %%
# Determining number of labels each
print('Fruit class distribution: \n',df['Label'].value_counts())

#%%[markdown]
## Expected Results of Clustering the data.
# It is known that our favorite orange colored fruits oranges, mandarins, and tarngerines are more similar than lemons and limes. This is due to pH and probably other molecular reasons. Therefore, when we transform the data and run a PCA we would see a clustering of the orange colored fruits and a small separation from lemons and limes. Below a PCA will be shown of this.
#

# %%
# Displaying the PCA Plot

# Encode the labels
df_n = df.copy()
le = LabelEncoder()
y = le.fit_transform(df_n['Label'])

# Transform data by Z-scale transformation and PCA
scaler = StandardScaler()
df_scale = scaler.fit_transform(df_n.drop(['Label'], axis = 1))
pca = PCA(n_components=3)
df_pca = pca.fit_transform(df_scale)

print('As you can see, the three orange colored fruits are more clustered closely than the lemons and limes. This indicates that there are enough differences to possibly perform an anomaly detection.')
# Plot the results
colors = ['r', 'g', 'b', 'c', 'm']
for i in range(5):
    plt.scatter(df_pca[y == i, 1], df_pca[y == i, 2], c=colors[i], label=le.classes_[i])
plt.legend()
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

#%%[markdown]
## The process for anomaly detection.
# The goal of this example is to first demonstrate an algorithm that can be used to correctly identify the orange colored fruits (OMT) from lemons and limes. What this means is, we are considering OMT measurments as non-anomolous that we want to further classify and then we are considering lemons and limes anomolous which are samples that are foreign to our modeling. Therefore, we want our model to discriminate OMT from lemons and limes which should not be further classified later. The first step here will be to separate out the data to train the model. To train the model we will take a subset of orange, mandarin and tangerine samples (df_omt). Then the remainder data under the variable df_ml will contain the remainder OMT samples plus the lemon and limes to validate the model.
#
# %%
# Separating out the data.
df_omt = df.loc[(df['Label'] == 'orange') | (df['Label'] == 'mandarin') |
                (df['Label'] == 'tangerine')]
df_ml = df.loc[(df['Label'] == 'lemon') | (df['Label'] == 'lime')]

print('df_omt only has:', df_omt['Label'].unique())
print('df_ml so far only has:', df_ml['Label'].unique())

# %%
# Splitting the OMT data into subsets for training and validation, retaining 80% for training
x_train, x_temp, y_train, y_temp = train_test_split(df_omt.iloc[:, 0:12],
                                                    df_omt['Label'].squeeze(), test_size= 0.2,
                                                    shuffle = True,
                                                    random_state=0)
print('Train Dimensions (measurements, ns): {}, Temporary Dimensions (measurements, ns): {}'.format(x_train.shape, x_temp.shape))

# %%
# Appending remiander 20% OMT back to df_ml to make validation set.
x_temp['Label'] = y_temp
df_new = pd.concat([df_ml, x_temp]) # Merge data files.
df_new.info()
df_new.reset_index(drop = True, inplace = True)
print('')
print('Validation set sample distribution: \n',df_new['Label'].value_counts())


# %%
#Separating out x and y values for model validation
x_test = df_new.iloc[:, 0:12]
y_test = df_new['Label']

# %%
# Transforming the training data by fitting a scaler and pca model.
scaler = MinMaxScaler()
pca = PCA(n_components=5, whiten=False)

x_train_scaled = scaler.fit_transform(x_train)
x_train_pca = pca.fit_transform(x_train_scaled)

#%%[markdown]
## Fitting the Anomaly detection model.
#
# Now that the training data is prepared, a One-class SVM will be trained on the 80% subset OMT samples. The model considers non-anomolous data as "1" and anomalies detected as "-1" as will be seen in the test prediction set (x_test).
#
# %%
# Perform anomaly detection training using One-class SVM
ocsvm = OneClassSVM(kernel='rbf', gamma=0.01, nu=0.1)
ocsvm.fit(x_train_pca)

# %%
# Transform test set and predict
x_test_scaled = scaler.transform(x_test)
x_test_pca = pca.transform(x_test_scaled)
anomalies = ocsvm.predict(x_test_pca)
print('Predicted output of anomaly detection on test samples', anomalies)

# %%
# Cleaning y_test and setting to a new value of y_convert.
y_convert = [] # Empty list to take y_test values and denoting them as 1 for OMT samples of -1 for lemon and lime.
for i in range(0, len(y_test)):
    if (y_test[i] == 'orange' or y_test[i] == 'mandarin' or y_test[i] == 'tangerine'):
        y_convert.append(1)
    
    else:
        y_convert.append(-1)
print('Show distribtution of anomaly values:', y_convert)

#%%[markdown]
## Prediction results of Anomaly detection.
# Below is the classification report and confusion matrix of the anomaly detection. The model overall performed at 85% accuracy. What is clear is that the model has a higher false positive rate than false negative rate given that it is less precise, predicting more anomolous samples (-1) as non-anomolous (1). Nonetheless, this is an example and we will continue to step 2 which is now the multi-classification conversion.
#
# %%
# Show classification report
print(classification_report(np.array(y_convert), np.array(anomalies)))
disp = ConfusionMatrixDisplay(confusion_matrix(y_convert, anomalies, labels = [1, -1]), display_labels = [1, -1])
disp.plot()
plt.show()
#%%[markdown]
## Developing Multi-classification Approach.
# Now that we have developed the first step which was anomaly detection. Now we will perform a mutliclassification for only OMT samples. The purpose of the anomaly detection was to separate out non OMT samples for future prediction of OMT samples only. Though we will see some contamination of non-OMT samples in prediction, this is to demonstrate the proof-of-principal. For multiclassification, we will use a strategy of a StandScaling transformation, followed by a PCA over 5 components which will then be modeled on a SVM with no parameterization for simplicity sake.

#

# %%
# Preparing the training data and model.
SM = SVC()
SM.fit(X=x_train_pca, y=y_train)

# Preparing test data based on anomaly detection output
df_new['Anomaly#'] = anomalies # Set new column factoring anomaly output, then separate for only values that are 1.
df_non_anom = df_new.loc[df_new['Anomaly#'] == 1] # Keep only non-anomolous
df_non_anom.drop(['Anomaly#'], inplace = True, axis = 1)
x_anom = df_non_anom.iloc[:, 0:12] #Selecting x data for nanosenosrs
y_anom = df_non_anom['Label']

x_anom_scaled = scaler.transform(x_anom)
x_anom_pca = pca.transform(x_anom_scaled)

y_pred = SM.predict(x_anom_pca)


#%%[markdown]
## Interpreting multiclassification prediction.
# Now that the modeling and prediction is done. See below for results. Overall, the modeling performed at below 70% accuracy. The reason for this is that the model was not trained for lemon data which should have been filtered during the anomaly detection step. This should demonstrate the importance of using an anomaly detection approach to identify unknowns in a system that our model is not trained for. Though the predictive accuracy is poor, this is an example dataset using different fruits and the modeling parameters were not tuned. If effort were to be placed for tuning parameters, filtering out nanosenosrs, etc. then the accuracy might be able to be improved. 
# %%
#Providing classification metrics
print(classification_report(np.array(y_anom), np.array(y_pred)))
disp = ConfusionMatrixDisplay(confusion_matrix(y_anom, y_pred, labels = df_non_anom['Label'].unique()), display_labels = df_non_anom['Label'].unique())

disp.plot()
plt.show()
