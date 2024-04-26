#1
import csv
a= []

with open('/content/sample_data/play.csv','r') as csvfile:
    for row in csv.reader(csvfile):
        a.append(row)
    print(a)

print("Total instance:\n",len(a))
num_attribute=len(a[0])-1
hypothesis = ['0']*num_attribute
print(hypothesis)
for i in range(0, len(a)):
    if a[i][num_attribute] == 'yes':
        for j in range(0,num_attribute):
            if hypothesis[j] == '0' or hypothesis[j] == a[i][j]:
                hypothesis[j] = a[i][j]
            else:
                hypothesis[j] = '?'
    print("\n The hypothesis for the training instance {} is :\n". format(i+1), hypothesis)

print("\n",hypothesis)

#2
import csv
with open("/content/sample_data/play.csv") as f:
    csv_file = csv.reader(f)
    data = list(csv_file)
    s = data[1][:-1]
    g = [['?' for _ in range(len(s))] for _ in range(len(s))]
print("Initial Specific Hypothesis:", s)
print("Initial General Hypothesis:", g)

for index, i in enumerate(data):
    if i[-1] == "yes":
        for j in range(len(s)):
            if i[j] != s[j]:
                s[j] = '?'
                g[j][j] = '?'
    elif i[-1] == "no":
        for j in range(len(s)):
            g[j][j] = '?' if s[j] == '?' else i[j]

    print("\nSteps of Candidate Elimination Algorithm", index + 1)
    print("Specific Hypothesis:", s)
    print("General Hypothesis:", g)

gh = []
for i in g:
  for j in i:
    if j!='?':
      gh.append(i)
      break

print("\nFinal Specific Hypothesis:\n", s)
print("\nFinal General Hypothesis:\n", gh)

#3
from efficient_apriori import apriori
import pandas as pd

store = pd.read_csv('/content/sample_data/apriori.csv', names=['new'], header=None)

print(store, "\n")

transactions = list(store['new'].apply(lambda x: x.split(",")))

itemsets, rules = apriori(transactions, min_support=0.3, min_confidence=0.6)

for i in itemsets:
    split_dicts = [{item: support} for item, support in itemsets[i].items()]
    for d in split_dicts:
        itemset_str = ','.join(list(d.keys())[0])
        support = list(d.values())[0]
        print("\n{:<20} {:15}".format(itemset_str, support))

print("\n{:<20}{:<25}{:<15}{:<15}{:<15}".format("Antecedent(lhs)", "Consequent(rhs)", "Support", "Confidence", "Lift"))
for rule in rules:
    if rule.support >= 0.3:
        print("{:<20}==>{:<20}{:<15.4f}{:<15.4f}{:<10.4f}".format(str(rule.lhs), str(rule.rhs), rule.support, rule.confidence, rule.lift))

#4a
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = pd.read_csv('Hours and Scores.csv')

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with CSV Dataset')
plt.show()

print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)

#4b
import numpy
from sklearn import linear_model

X = numpy.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1,1)
y = numpy.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

logr = linear_model.LogisticRegression()
logr.fit(X,y)

predicted = logr.predict(numpy.array([4.52]).reshape(-1,1))
print(predicted)

train_acc = logr.score(X, y)
print("The Accuracy for Training Set is {}".format(train_acc*100))

#5
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

df = pd.read_csv('/content/sample_data/id.csv')
df.fillna(method='ffill', inplace=True)
df = pd.get_dummies(df, columns=['Outlook', 'Temperature', 'Humidity', 'Wind'])
X = df.drop(columns=['Play'])
y = df['Play']

model = DecisionTreeClassifier()
model.fit(X, y)

tree_rules = export_text(model, feature_names=list(X.columns))
print(tree_rules)
