import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from DeepFM import SparseFeat, DenseFeat, DeepFM

data = pd.read_csv('movie_data.csv')
target = ['rating']
for s in data.columns:
    new_str = s.replace(' ', '_')
    data.rename(columns={s: new_str}, inplace=True)

sparse_features = data.columns[13:45]
dense_features = [col for col in data.columns if col not in list(sparse_features) + target]

for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])
mms = MinMaxScaler(feature_range=(0, 1))
data[data.columns] = mms.fit_transform(data[data.columns])

fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique()) for feat in sparse_features] + [DenseFeat(feat, 1, ) for feat in dense_features]
linear_feature_columns = fixlen_feature_columns
dnn_feature_columns = [DenseFeat(feat, 1, ) for feat in dense_features]
feature_names = dense_features = [col for col in data.columns if col not in target]

# 3.generate input data for model
train, test = train_test_split(data, test_size=0.2)
train_model_input = {name: train[name] for name in feature_names}
test_model_input = {name: test[name] for name in feature_names}

# 4.Define Model,train,predict and evaluate
device = 'cpu'
use_cuda = True
if use_cuda and torch.cuda.is_available():
    print('cuda ready...')
    device = 'cuda:0'

model = DeepFM(linear_feature_columns, dnn_feature_columns, device=device)

history = model.fit(train_model_input, train[target].values, batch_size=256, epochs=5, validation_split=0.2)
pred_ans = model.predict(test_model_input, batch_size=256)
print("test MSE", round(mean_squared_error(test[target].values, pred_ans), 4))
