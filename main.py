# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import pickle

#use pandas to load data:
# Load dataset
url = "/home/ritvik/Desktop/FlowerML/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

#dimensions of dataset
#how many instances (rows) and how many attributes (columns) the data contains with shape property
print(dataset.shape)

#peek at data:
print(dataset.head(20))

#statistical summary (count, mean, min, max and percentiles)
print(dataset.describe())

#class distribution (number of instances that belong to eas class)
print(dataset.groupby('class').size())

#univaraibe plots helps us understand each attribute
#box and whisker plot:
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()

# histograms
dataset.hist()
pyplot.show()

#multivariate plots let us look at interactions between variables
#allows us to spot structured realtionships between input variables
# scatter plot matrix
scatter_matrix(dataset)
pyplot.show()

#Create a validation data set
# We will split the loaded dataset into two, 80% of which we will use to train, evaluate and select among our models, and 20% that we will hold back as a validation dataset.
# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.20, random_state=1)

#test harness
#we will use stratified 10-fold cross validation to estimate model accuracy
#splits dataset into 10 parts, train on 9 and test on 1 and repeat for all combiantions of train-test splits
#stratified means that each fold or split of dataset will aim to ahve the same distribution of example by class as exist in the whole training dataset

# Spot Check Algorithms (test to see which one is most accurate for this data)
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn
results = []
names = []
for name, model in models:
    # Test options and evaluation metric
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Compare Algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

# Make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

# Evaluate predictions
print(accuracy_score(Y_validation, predictions)) # gives accuracy
print(confusion_matrix(Y_validation, predictions)) # gives errors made
print(classification_report(Y_validation, predictions)) #breakdown of each class by precision
