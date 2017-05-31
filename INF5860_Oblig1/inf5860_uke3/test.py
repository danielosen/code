import SFCrime as sfc
data = sfc.get_data(categories=['RUNAWAY', 'TRESPASS'])
ax = sfc.plot_map(data)
from classification import NearestNeighborClassification

clf = NearestNeighborClassification()

x_train, x_test, y_train, y_test = sfc.data_to_train_test(data)
print(y_test)
clf.fit(x_train, y_train)

y_ = clf.predict(x_test)
print y_[0:10], y_test[0:10]
print 'Accuracy', (y_ == y_test).mean()