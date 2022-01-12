import os                      
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import sklearn
from sklearn import linear_model, metrics, preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler,OrdinalEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import r2_score, f1_score


from sklearn.model_selection import train_test_split
# --------cross-validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
# -------- classification
import sklearn
from sklearn import neighbors, tree, ensemble, naive_bayes, svm
# *** KNN
from sklearn.neighbors import KNeighborsClassifier
# *** Decision Tree; Random Forest
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# *** Naive Bayes
from sklearn.naive_bayes import GaussianNB
# *** SVM classifier
from sklearn.svm import SVC
# --------  metrics:
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import make_scorer
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt


def load_dataset(file_name, target_column): 
    df=pd.read_csv(file_name)
    y=df[target_column]
    z=df.loc[:,['Player','Team','League','Version','POS','W/Rs','Foot','Stats']]
    X=df.drop([target_column,'Player','Team','League','Version','POS','W/Rs','Foot','Stats','PAC','SHO','PAS','DRI','OVR','PHY','DEF'],axis=1)
    
    
    X["special_cardsXtime"]=X['special cards']*X['time']

    X["pos_goals"]=X['Pos-label']*X['goals'] #*X['assists']
    X["KEY_AS"]=X['key_passes']*X['assists']
    X["x"]=X['goals']*X['shots']
    #X["x1"]=X['Pos-label']*X['Team Goals against']
    #X["x2"]=X['DEF']*X['Pos-label']
  
  
    return X,y,z


def calc_evaluation_val(eval_metric, y_test, y_predicted):
    if (eval_metric=='accuracy'):
        a= metrics.accuracy_score(y_true = y_test, y_pred = y_predicted)
        return a
    if (eval_metric=='precision'):
        a= metrics.precision_score(y_true = y_test, y_pred = y_predicted)
        return a
    if (eval_metric=='recall'):
        a= metrics.recall_score(y_true = y_test, y_pred = y_predicted)
        return a
    if (eval_metric=='f1'):
        a= metrics.f1_score(y_true = y_test, y_pred = y_predicted)
        return a
    if (eval_metric=='confusion_matrix'):
        a= metrics.confusion_matrix(y_true = y_test, y_pred = y_predicted)
        return a


def get_classifier_obj(classifier_name, params):
    if(classifier_name=='KNN'):
        if(params==None):
            clf=KNeighborsClassifier()
        else:
            clf=KNeighborsClassifier(params['n_neighbors'])
        return clf
    if(classifier_name=='decision_tree'):
        if(params==None):
        
            clf=DecisionTreeClassifier()
        else:
            max_min_val=list(params.values())
            clf=DecisionTreeClassifier(max_depth=max_min_val[0],min_samples_split=max_min_val[1])
        return clf
    if(classifier_name=='random_forest'):
        if(params==None):
            clf=RandomForestClassifier()
        else:
            clf=RandomForestClassifier(params['n_estimators'])
            
        return clf
    if(classifier_name=='svm'):
        clf=SVC()
        return clf
    if(classifier_name=='naive_bayes'):
        clf=GaussianNB()
        return clf


file_name='tFIFA 20 + Statistical data 2020.csv'

target_col_name = 'tots'
X_train, y_train ,z_train= load_dataset(file_name, target_col_name)



#logistic regression
#m=linear_model.LinearRegression().fit(X_train,y_train)
lrm=linear_model.LogisticRegression().fit(X_train,y_train)
file_name_test='tFIFA 21 + Statistical data 2021.csv'
X_test,y_test,z_test=load_dataset(file_name_test, target_col_name)

y_predicted=lrm.predict(X_test)

accuracy_val = calc_evaluation_val("accuracy", y_test, y_predicted)
precision_val = calc_evaluation_val("precision", y_test, y_predicted)
recall_val = calc_evaluation_val("recall", y_test, y_predicted)
f1_val = calc_evaluation_val("f1", y_test, y_predicted)
confusion_matrix_val = calc_evaluation_val("confusion_matrix", y_test, y_predicted)

print("accuracy is:",metrics.accuracy_score(y_test, y_predicted))
print("precision is:",metrics.precision_score(y_test, y_predicted))
print("recall is:",metrics.recall_score(y_test, y_predicted))
print("f1 is:",metrics.f1_score(y_test, y_predicted))
print("Confusion metrix is: \n " , confusion_matrix_val)


#print(mat)
df=pd.read_csv(file_name_test)
df['y_test']=y_test.tolist()
df['y_predicted']=y_predicted.tolist()






#trying some algorithmes
params_knn = {'n_neighbors':10}
params_random_forest = {'n_estimators':51}
params_decision_tree = {'max_depth':6, 'min_samples_split':6}
print(list(params_decision_tree.values()))
clf_naive_bayes = get_classifier_obj("naive_bayes",None)
clf_svm = get_classifier_obj("svm",None)
clf_knn = get_classifier_obj("KNN",None)
clf_random_forest = get_classifier_obj("random_forest",None)
clf_decision_tree = get_classifier_obj("decision_tree",None)
clf_knn_with_params = get_classifier_obj("KNN",params_knn)
clf_random_forest_with_params = get_classifier_obj("random_forest",params_random_forest)    
clf_decision_tree_with_params = get_classifier_obj("decision_tree",params_decision_tree)


#chosen algorithem 
model=clf_naive_bayes.fit(X_train,y_train)
y_predicted=model.predict(X_test)






accuracy_val = calc_evaluation_val("accuracy", y_test, y_predicted)
precision_val = calc_evaluation_val("precision", y_test, y_predicted)
recall_val = calc_evaluation_val("recall", y_test, y_predicted)
f1_val = calc_evaluation_val("f1", y_test, y_predicted)
confusion_matrix_val = calc_evaluation_val("confusion_matrix", y_test, y_predicted)

df['y_test']=y_test.tolist()
df['y_predicted']=y_predicted.tolist()

print("accuracy is:",metrics.accuracy_score(y_test, y_predicted))
print("precision is:",metrics.precision_score(y_test, y_predicted))
print("recall is:",metrics.recall_score(y_test, y_predicted))
print("f1 is:",metrics.f1_score(y_test, y_predicted))
print("Confusion metrix is: \n " , confusion_matrix_val)



fig, ax = plt.subplots(1, 2, figsize=(10, 3), sharey=False)
plot_confusion_matrix( clf_naive_bayes, X_test, y_test,ax=ax[0])  
ax[0].set_title('confusion matrix for logistic regression model')

plot_confusion_matrix(lrm, X_test, y_test,ax=ax[1])  
ax[1].set_title('confusion matrix for naive bayes model')


plt.show()