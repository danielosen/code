import pandas as pd
import numpy as np
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns

# Supplied map bounding box:
#    ll.lon     ll.lat   ur.lon     ur.lat
#    -122.52469 37.69862 -122.33663 37.82986

mapdata = np.loadtxt("./input/sf_map_copyright_openstreetmap_contributors.txt")
asp = mapdata.shape[0] * 1.0 / mapdata.shape[1]

lon_lat_box = (-122.5247, -122.3366, 37.699, 37.8299)
clipsize = [[-122.5247, -122.3366],[ 37.699, 37.8299]]

def get_data(var='Category', categories=['RUNAWAY', 'TRESPASS']):
  z = zipfile.ZipFile('./input/train.csv.zip')
  data = pd.read_csv(z.open('train.csv'))
  print 'Categories:', np.unique(data.Category)

  #Get rid of the bad lat/longs
  data['Xok'] = data[data.X<-121].X
  data['Yok'] = data[data.Y<40].Y
  data = data.dropna()
  train_cat = data[np.in1d(data.Category, categories)]
  cat_size_min = train_cat.groupby(var).size().min()
  return pd.concat([a[1].sample(cat_size_min) for a in train_cat.groupby(var)])



def data_to_train_test(data):
  from sklearn.model_selection import train_test_split
  x, y = data[['Xok', 'Yok']].as_matrix(), data['Category'].as_matrix()
  return train_test_split(x, y, test_size=0.66)

def plot_map(data, var='Category'):
  colors = ['Reds', 'Blues', 'Purples', 'Greens']
  plt.figure(figsize=(20,20*asp))
  ax = plt.gca()
  for i, (category, df) in enumerate(data.groupby(var)):
    a = sns.kdeplot(df.Xok, df.Yok, clip=clipsize, aspect=1/asp, cmap=colors[i], ax=ax, legend=True, label=category)

  ax.imshow(mapdata, cmap=plt.get_cmap('gray'),
                extent=lon_lat_box,
                aspect=asp)
  plt.show()
  return ax

if __name__ == '__main__':
  data = get_data()
  data_to_train_test(data)
  plot_map(data)
