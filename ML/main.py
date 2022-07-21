# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    data = pd.read_csv("D:\\Study\\LABS\\NIR\\SQLi_detection_ML\\Data\\Centrality_Measure_Dataset.csv")
    print(data.head())
    training_set, test_set = train_test_split(data, test_size=0.80, random_state=4223244)
    print("train:", training_set.head())
    print("test:",test_set.head())

    x_train = training_set.iloc[:, 0:164].values  # data
    y_train = training_set.iloc[:, 164].values  # target
    x_test = test_set.iloc[:, 0:164].values  # data
    y_test = test_set.iloc[:, 164].values  # target

    print(x_train, y_train)
    print(x_test, y_test)

    #обучаем модель
    #classifier = SVC(kernel='linear')
    classifier = SVC(kernel='rbf',random_state=1,C=1,gamma=0.001)
    classifier.fit(x_train, y_train)

    # делаем предикшн на x_test
    y_pred = classifier.predict(x_test)
    print("y_pred:",y_pred)

    # матрица ошибок и точность
    cm = confusion_matrix(y_test, y_pred)
    print('cm:', cm)
    accuracy = float(cm.diagonal().sum()) / len(y_test)
    print('accuracy score:', metrics.accuracy_score(y_test, y_pred) * 100, '%')
    print('precision score:', metrics.precision_score(y_test, y_pred) * 100, '%')
    print('recall score:', metrics.recall_score(y_test, y_pred) * 100, '%')

    tpr = cm[0][0] / (cm[0][0] + cm[0][1]) * 100
    fpr = cm[1][0] / (cm[1][0] + cm[1][1]) * 100
    print('TPR is:', tpr,
          '%')  # процент среди всех positive верно предсказан моделью
    print('FPR is:', fpr,
          '%')  # процент среди всех negative неверно предсказан моделью

    positive_pred = y_pred
    print('positive pred:', positive_pred)

    svm_auc = roc_auc_score(y_test, positive_pred)
    print('SVM: ROC AUC=%.3f' % (svm_auc))

    # рассчитываем roc-кривую
    fpr, tpr, treshold = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    # строим график
    plt.plot(fpr, tpr, color='darkorange',
             label='ROC кривая (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Пример ROC-кривой')
    plt.legend(loc="lower right")
    plt.show()

