from sklearn import svm
from sklearn.model_selection import train_test_split as tts
from sklearn import datasets

wine = datasets.load_wine()


features = wine.data
labels = wine.target

clf = svm.SVC(kernel= 'linear')

train_feats, test_feats, train_labels, test_labels = tts(features, labels, test_size = 0.2)

clf.fit(train_feats, train_labels)

predictions = clf.predict(test_feats)
print(predictions)

count = 0
for i in range(len(predictions)):
    if(predictions[i] == test_labels[i]):
        count+=1
print("ACCURACY", (count/len(predictions)*100))
