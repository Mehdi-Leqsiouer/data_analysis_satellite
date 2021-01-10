import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from flask import Flask, request

app = Flask(__name__)

def fonction_princ():
    #Retrieve and convert training data into dataframe
    training_set = pd.read_csv('sat.trn', delimiter=' ', nrows = None, header = None)
    training_set.dataframeName = 'training_set'
    training_set.columns = [*training_set.columns[:-1], 'Y']
    nRow, nCol = training_set.shape
    print(training_set.shape)
    training_set.head(5)

    #Retrieve and convert testing data into dataframe
    testing_set = pd.read_csv('sat.tst', delimiter=' ', nrows = None, header = None)
    testing_set.dataframeName = 'testing_set'
    testing_set.columns = [*testing_set.columns[:-1], 'Y']
    nRow, nCol = testing_set.shape
    print(testing_set.shape)
    testing_set.head(5)

    X_train = training_set.iloc[:, :-1]
    y_train = training_set["Y"]
    X_test = testing_set.iloc[:, :-1]
    y_test = testing_set["Y"]
    
    fig, ax = plt.subplots(figsize=(24,5))
    pd.DataFrame(training_set.iloc[-1400:,1:-1].reset_index(drop=True)).plot(ax=ax, legend=False)
    fig, ax = plt.subplots(figsize=(24,3))
    training_set.loc[-1400:,"Y"].reset_index(drop=True).plot(ax=ax)

    #Create a Gaussian Classifier
    classifier=RandomForestClassifier(n_estimators=100)

    #Train the model using the training sets y_pred=clf.predict(X_test)
    classifier.fit(X_train,y_train)
    
    #Prediction
    y_pred=classifier.predict(X_test)
    
    # Model Accuracy
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))  
    
    return classifier

@app.route("/", methods = ['POST'])
def prediction_api():
    classifier = fonction_princ()
    pute = classifier.predict([request.json.get("value")])
    return {"value": int(pute[0])}


if __name__ == '__main__':
    app.run(debug=True)