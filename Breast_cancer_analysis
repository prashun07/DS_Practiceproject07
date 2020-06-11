#importing libraries................./
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
#loading Datasets............./
cancer_data=datasets.load_breast_cancer()
x_train,x_test,y_train,y_test=train_test_split(cancer_data.data,cancer_data.target)
cls=svm.SVC(kernel="linear") #C-Support Vector Classification.
cls.fit(x_train,y_train)
pred=cls.predict(x_test)
print("accuracy:",metrics.accuracy_score(y_test,y_pred=pred))#Accuracy classification score.
print("Precision:",metrics.precision_score(y_test,y_pred=pred))#Compute the precision
print("Recall:",metrics.recall_score(y_test,y_pred=pred))#Compute the recall
print(metrics.classification_report(y_test,y_pred=pred)) #Build a text report showing the main classification metrics.


#***************OUTPUT***************************
'''accuracy: 0.9440559440559441
Precision: 0.9333333333333333
Recall: 0.9767441860465116
              precision    recall  f1-score   support

           0       0.96      0.89      0.93        57
           1       0.93      0.98      0.95        86

    accuracy                           0.94       143
   macro avg       0.95      0.94      0.94       143
weighted avg       0.94      0.94      0.94       143'''

