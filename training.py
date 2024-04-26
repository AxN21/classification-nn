from model import DNN
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets, metrics, model_selection, preprocessing
from sklearn.impute import SimpleImputer

def binary_dnn_model_results(model, X_train, Y_train, X_test, Y_test):
    print ('DNN model results')

    # LEARNING PHASE
    print('GRADIENT DESCENT CHECK')
    plt.plot(model.cost_during_training)
    plt.title('Cost value during training')
    plt.xlabel('Iteration')
    plt.ylabel('Croos Entropy Cost')
    plt.show()

    # ACCURACY AND ROC ON TRAIN DATA
    print('\nTRAINING METRICS')
    train_preds = model.predict(X_train)
    train_preds_labels = train_preds > 0.5
    train_accuracy = np.sum(train_preds_labels == Y_train) / Y_train.shape[1]
    print(f'Train Accuracy: {train_accuracy}')
    train_roc_auc_score = metrics.roc_auc_score(Y_train.T, train_preds.T)
    print(f'Train ROC AUC SCORE {train_roc_auc_score}')

    # ACCURACY AND ROC ON TEST DATA
    print('\nTEST MATRICS')
    test_preds = model.predict(X_test)
    test_preds_labels = test_preds > 0.5
    test_accuracy = np.sum(test_preds_labels == Y_test) / Y_test.shape[1]
    print(f'Test Accuracy: {test_accuracy}')
    test_roc_auc_score = metrics.roc_auc_score(Y_test.T, test_preds.T)
    print(f'Test ROC AUC SCORE {test_roc_auc_score}')

iris = datasets.load_iris()


X = np.array(iris.data[:100])
Y = np.array(iris.target[:100])
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=42)

X_train = X_train.T
Y_train = Y_train.reshape(1, len(Y_train))
X_test = X_test.T
Y_test = Y_test.reshape(1, len(Y_test))

print(X_train.shape)
print(X_test.shape)

model = DNN()
model.train(X_train=X_train, Y_train=Y_train, layer_dims=[5, 4, 3, 1], epoch=1_000)
binary_dnn_model_results(model, X_train, Y_train, X_test, Y_test)
