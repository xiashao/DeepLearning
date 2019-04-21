from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import tree
from sklearn import preprocessing
from sklearn.externals.six import StringIO

# Read in the csv file and put features into list of dict and list of class label
allElectronicsData = open(r'D:\Users\DELL\workspace\DTree\AllElectronics.csv', 'rb')
reader = csv.reader(allElectronicsData)
headers = reader.next()

print(headers)

featureList = []
labelList = []

for row in reader:
    labelList.append(row[len(row)-1])
    rowDict = {}
    for i in range(1, len(row)-1):
        rowDict[headers[i]] = row[i]
    featureList.append(rowDict)

print(featureList)

vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()
print(dummyX)
print(vec.get_feature_names())

lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print(dummyY)

clf=tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX,dummyY)
print(clf)


with open("allElectInfo.dot","w") as f:
    f=tree.export_graphviz(clf,feature_names=vec.get_feature_names(),out_file=f)
    
oneRow=dummyX[0,:].reshape(1,-1)
print(oneRow)

newRow=oneRow
newRow[0][0]=1
newRow[0][2]=0
newRow=newRow.reshape(1,-1)
print("newrowis" + str(newRow))
predictY=clf.predict(newRow)
print(predictY)

