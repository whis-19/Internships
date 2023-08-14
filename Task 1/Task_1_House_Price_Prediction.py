
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas._libs import missing
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error


from scipy import stats


import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

pd.options.display.max_rows = 100
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_sample = pd.read_csv('sample_submission.csv')






sns.histplot(df_train['SalePrice'])
#plt.show()
sns.scatterplot(df_train,x='GrLivArea',y='SalePrice')
# plt.show()
sns.scatterplot(df_train,x='LotArea',y='SalePrice')
# plt.show()
df = df_train.append(df_test)

# print(df)

missing_data = df.isna().sum().sort_values(ascending=False)

df['PoolQC']=df['PoolQC'].fillna('No Pool')
df['Alley']=df['Alley'].fillna('No Alley')
df['BsmtQual']=df['BsmtQual'].fillna('No Basement')
df['MiscFeature']=df['MiscFeature'].fillna('No Misc')
df['FireplaceQu']=df['FireplaceQu'].fillna('No Fireplace')
df['GarageFinish']=df['GarageFinish'].fillna('No Garage')
df['GarageQual']=df['GarageQual'].fillna('No Garage')
df['GarageCond']=df['GarageCond'].fillna('No Garage')
df['GarageType']=df['GarageType'].fillna('No Garage')
df['BsmtExposure']=df['BsmtExposure'].fillna('No Basement')
df['BsmtCond']=df['BsmtCond'].fillna('No Basement')
df['BsmtFinType1']=df['BsmtFinType1'].fillna('No Basement')
df['BsmtFinType2']=df['BsmtFinType2'].fillna('No Basement')
df['MasVnrType']=df['MasVnrType'].fillna('No Masvnr')

df['MSZoning'] = df['MSZoning'].fillna(df['MSZoning'].mode()[0])
df['Utilities'] = df['Utilities'].fillna(df['Utilities'].mode()[0])
df['Functional'] = df['Functional'].fillna(df['Functional'].mode()[0])
df['Exterior1st'] = df['Exterior1st'].fillna(df['Exterior1st'].mode()[0])
df['Exterior2nd'] = df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0])
df['SaleType'] = df['SaleType'].fillna(df['SaleType'].mode()[0])
df['KitchenQual'] = df['KitchenQual'].fillna(df['KitchenQual'].mode()[0])
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

df['LotFrontage'] = df['LotFrontage'].fillna(df['LotArea']**0.5)
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])


df['MasVnrArea'] = df['MasVnrArea'].fillna(df['MasVnrArea'].median())
df['BsmtFullBath'] = df['BsmtFullBath'].fillna(df['BsmtFullBath'].mode()[0])
df['BsmtHalfBath'] = df['BsmtHalfBath'].fillna(df['BsmtHalfBath'].mode()[0])
df['GarageCars'] = df['GarageCars'].fillna(df['GarageCars'].mode()[0])
df['GarageArea'] = df['GarageArea'].fillna(df['GarageArea'].median())
df['TotalBsmtSF'] = df['TotalBsmtSF'].fillna(df['TotalBsmtSF'].median())
df['BsmtUnfSF'] = df['BsmtUnfSF'].fillna(df['BsmtUnfSF'].median())
df['BsmtFinSF2'] = df['BsmtFinSF2'].fillna(df['BsmtFinSF2'].median())
df['BsmtFinSF1'] = df['BsmtFinSF1'].fillna(df['BsmtFinSF1'].median())

missing_data = df.isna().sum().sort_values(ascending=False)


df = df.replace({"MSSubClass":{20:"SC20", 30:"SC30", 40:"SC40",
                               45:"SC45", 50:"SC50", 60:"SC60",
                               70:"SC70", 75:"SC75", 80:"SC80",
                               85:"SC85", 90:"SC90", 120:"SC120",
                               150:"SC150", 160:"SC160",
                               180:"SC180", 190:"SC190"},
                 "MoSold":{1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr",
                           5:"May", 6:"Jun", 7:"Jul", 8:"Aug",
                           9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"}})


df['HeatingQC'] = df['HeatingQC'].replace('Po', 'Fa')

def season_winter(row):
  if row['MoSold'] in ['Dec', 'Jan', 'Feb']:
    return 1
  else:
    return 0

df['Winter'] = df.apply(lambda row: season_winter(row), axis = 1).astype('int8')

def winter_poor_heating_dummy(row):
  if (row['Winter'] == 1) & (row['HeatingQC'] == 'Fa'):
    return 1
  else:
    return 0

df['Winter_poor_heating'] = df.apply(lambda row: winter_poor_heating_dummy(row),
                                     axis = 1).astype('int8')


sns.catplot(df, x = 'HeatingQC', y = 'SalePrice', col = 'Winter', kind = 'bar', estimator = 'median')

# plt.show()

fig, axes = plt.subplots(1, 2, figsize = (10, 5), sharey = True)
sns.scatterplot(df, x='OverallQual', y='SalePrice', ax = axes[0])
sns.scatterplot(df, x='OverallCond', y='SalePrice', ax = axes[1])
#plt.show()

df.groupby('OverallCond')['SalePrice'].agg(['count', 'mean'])

def dummy_condition(row):
  if row['OverallCond'] < 5:
    return 1
  else:
    return 0

df['Poor_condition'] = df.apply(lambda row: dummy_condition(row),
                                axis = 1).astype('int8')

def dummy_extercondition(row):
  if row['ExterCond']in ['Fa', 'Po']:
    return 1
  else:
    return 0

df['Poor_exter_condition'] = df.apply(lambda row: dummy_extercondition(row),
                                axis = 1).astype('int8')


def numerical_encoding(row, col_name):
  if row[col_name] == 'Ex':
    return 5
  elif row[col_name] == 'Gd':
    return 4
  elif row[col_name] == 'TA':
    return 3
  elif row[col_name] == 'Fa':
    return 2
  elif row[col_name] == 'Po':
    return 1
  else:
    return 0

df['HeatingQC'] = df.apply(lambda row: numerical_encoding(row, 'HeatingQC'), axis = 1).astype('int8')
df['KitchenQual'] = df.apply(lambda row: numerical_encoding(row, 'KitchenQual'), axis = 1).astype('int8')
df['ExterQual'] = df.apply(lambda row: numerical_encoding(row, 'ExterQual'), axis = 1).astype('int8')
df['ExterCond'] = df.apply(lambda row: numerical_encoding(row, 'ExterCond'), axis = 1).astype('int8')
df['BsmtQual'] = df.apply(lambda row: numerical_encoding(row, 'BsmtQual'), axis = 1).astype('int8')
df['FireplaceQu'] = df.apply(lambda row: numerical_encoding(row, 'FireplaceQu'), axis = 1).astype('int8')

df['Condition'] = (df['OverallQual'] + df['OverallCond'] + df['ExterQual'] + \
                  df['ExterCond'] + df['KitchenQual'])/5

Location = pd.qcut(df.groupby('Neighborhood')['Condition'].agg('mean'), 4, labels = ['bad', 'fair', 'good', 'excellent'])
df = pd.merge(df, Location.rename('Location'), how = 'left', on = 'Neighborhood')

df = df.drop(['Condition'], axis = 1)

fig, axes = plt.subplots(5, 3, figsize = (15, 15), sharey = True)
sns.boxplot(df, y = 'SalePrice', x = 'LotConfig', ax = axes[0,0])
sns.boxplot(df, y = 'SalePrice', x = 'Condition1', ax = axes[0,1])
sns.boxplot(df, y = 'SalePrice', x = 'PavedDrive', ax = axes[0,2])
sns.boxplot(df, y = 'SalePrice', x = 'Electrical', ax = axes[1,0])
sns.boxplot(df, y = 'SalePrice', x = 'BldgType', ax = axes[1,1])
sns.boxplot(df, y = 'SalePrice', x = 'HouseStyle', ax = axes[1,2])
sns.boxplot(df, y = 'SalePrice', x = 'SaleType', ax = axes[2,0])
sns.boxplot(df, y = 'SalePrice', x = 'Functional', ax = axes[2,1])
sns.boxplot(df, y = 'SalePrice', x = 'BsmtExposure', ax = axes[2,2])
sns.boxplot(df, y = 'SalePrice', x = 'PavedDrive', ax = axes[3,0])
sns.boxplot(df, y = 'SalePrice', x = 'RoofStyle', ax = axes[3,1])
sns.boxplot(df, y = 'SalePrice', x = 'MasVnrType', ax = axes[3,2])
sns.boxplot(df, y = 'SalePrice', x = 'Exterior1st', ax = axes[4,0])
sns.boxplot(df, y = 'SalePrice', x = 'Foundation', ax = axes[4,1])

#plt.show()

def cul_de_sac_dummy(row):
  if row['LotConfig'] == 'CulDSac':
    return 1
  else:
    return 0

df['Cul_de_sac'] = df.apply(lambda row: cul_de_sac_dummy(row), axis = 1).astype('int8')


def positive_feature_dummy(row):
  if (row['Condition1'] in ['PosN', 'PosA']) or (row['Condition2'] in ['PosN', 'PosA']):
    return 1
  else:
    return 0

df['Positive_feature'] = df.apply(lambda row: positive_feature_dummy(row), axis = 1).astype('int8')



def adjacent_main_road_dummy(row):
  if (row['Condition1'] in ['Artery', 'Feedr', 'RRAe']) or \
     (row['Condition2'] in ['Artery', 'Feedr', 'RRAe']):
    return 1
  else:
    return 0

df['Adjacent_main_road'] = df.apply(lambda row: adjacent_main_road_dummy(row), axis = 1).astype('int8')


def fuse_electrical_dummy(row):
  if (row['Electrical'] in ['FuseA', 'FuseF', 'FuseP', 'Mix']):
    return 1
  else:
    return 0

df['Fuse_electrical'] = df.apply(lambda row: fuse_electrical_dummy(row), axis = 1).astype('int8')


def shared_walls_dummy(row):
  if (row['BldgType'] in ['2FmCon', 'Duplx', 'Twnhs']):
    return 1
  else:
    return 0

df['Shared_walls'] = df.apply(lambda row: shared_walls_dummy(row), axis = 1).astype('int8')



def unfinish_dummy(row):
  if (row['HouseStyle'] in ['1.5Unf', '2.5Unf']):
    return 1
  else:
    return 0

df['Unfinish'] = df.apply(lambda row: unfinish_dummy(row), axis = 1).astype('int8')



def new_dummy(row):
  if (row['SaleType'] == 'New'):
    return 1
  else:
    return 0

df['New'] = df.apply(lambda row: new_dummy(row), axis = 1).astype('int8')


def deduction_dummy(row):
  if (row['Functional'] == 'Typ'):
    return 0
  else:
    return 1

df['Deduction'] = df.apply(lambda row: deduction_dummy(row), axis = 1).astype('int8')


def basement_exposure_dummy(row):
  if (row['BsmtExposure'] == 'Gd'):
    return 1
  else:
    return 0

df['Good_bsmt_exposure'] = df.apply(lambda row: basement_exposure_dummy(row),
                                        axis = 1).astype('int8')


def driveway_paved_dummy(row):
  if (row['PavedDrive'] == 'Y'):
    return 1
  else:
    return 0

df['Driveway_paved'] = df.apply(lambda row: driveway_paved_dummy(row),
                                        axis = 1).astype('int8')


def street_dummy(row):
  if (row['Street'] == 'Pave'):
    return 1
  else:
    return 0

df['Street_paved'] = df.apply(lambda row: street_dummy(row), axis = 1).astype('int8')


def hiproof_dummy(row):
  if row['RoofStyle'] == 'Hip':
    return 1
  else:
    return 0

df['Hip_roof'] = df.apply(lambda row: hiproof_dummy(row), axis = 1).astype('int8')


def vinyl_dummy(row):
  if row['Exterior1st'] == 'VinylSd':
    return 1
  else:
    return 0

df['Vinyl_ext'] = df.apply(lambda row: vinyl_dummy(row), axis = 1).astype('int8')


def stone_dummy(row):
  if row['MasVnrType'] == 'Stone':
    return 1
  else:
    return 0

df['Stone_masvnr'] = df.apply(lambda row: stone_dummy(row), axis = 1).astype('int8')


def concrete_dummy(row):
  if row['Foundation'] == 'PConc':
    return 1
  else:
    return 0

df['Concrete_foundation'] = df.apply(lambda row: concrete_dummy(row), axis = 1).astype('int8')


def air_numerical(row):
  if (row['CentralAir'] == 'Y'):
    return 1
  else:
    return 0

df['CentralAir'] = df.apply(lambda row: air_numerical(row), axis = 1).astype('int8')

df_train.groupby('YrSold')['SalePrice'].mean()

def one_hot_year(row):
  if row['YrSold'] >= 2008:
    return 1
  else:
    return 0

df['GFC'] = df.apply(lambda row: one_hot_year(row), axis = 1).astype('int8')


df['Age'] = df['YrSold'] - df['YearBuilt']

def label_remodel (row):
  if row['YearBuilt'] < row['YearRemodAdd'] :
    return 1
  else:
    return 0

df['Remodel'] = df.apply(lambda row: label_remodel(row), axis=1).astype('int8')

sns.scatterplot(df, x='Age', y='SalePrice', hue='Remodel')
#plt.shpw()

df['2ndFlr'] = pd.cut(df['2ndFlrSF'], bins = [-float('inf'), 0, float('inf')],
                      labels = [0, 1]).astype('int8')
df['Bsmt'] = pd.cut(df['TotalBsmtSF'], bins = [-float('inf'), 0, float('inf')],
                                  labels = [0, 1]).astype('int8')

fig, axes = plt.subplots(1, 2, figsize = (10, 5), sharey = True)
sns.scatterplot(df, x='GrLivArea', y='SalePrice', hue='2ndFlr', ax = axes[0])
sns.scatterplot(df, x='GrLivArea', y='SalePrice', hue='Bsmt', ax = axes[1])
#plt.show()



df['Pool'] = pd.cut(df['PoolArea'], bins = [-float('inf'), 0, float('inf')],
labels = [0, 1]).astype('int8')



sns.boxplot(df, x = 'Pool', y = 'SalePrice')
#plt.show()


columns_order = [col for col in df.columns if col != 'SalePrice'] + ['SalePrice']
df = df[columns_order]

correlation_matrix = df.select_dtypes(exclude = 'int8').corr()

plt.subplots(figsize = (10, 8))
sns.heatmap(correlation_matrix, vmax = 1)

variables = ['SalePrice', 'Id', 'HeatingQC', 'OverallQual', 'KitchenQual',
             'ExterQual', 'BsmtQual', 'Age', 'GrLivArea', 'BedroomAbvGr',
             'TotalBsmtSF', 'FireplaceQu', 'TotRmsAbvGrd', 'FullBath', 'HalfBath',
             'GarageCars', 'LotArea', 'Winter_poor_heating', 'Poor_condition',
             'Poor_exter_condition', 'Location',  'Cul_de_sac',
             'Positive_feature', 'Adjacent_main_road', 'Fuse_electrical',
             'Shared_walls', 'Unfinish', 'New', 'Deduction', 'Good_bsmt_exposure',
             'Driveway_paved', 'Street_paved', 'Hip_roof', 'Vinyl_ext',
             'Stone_masvnr', 'Concrete_foundation', 'CentralAir', 'GFC', 'Remodel',
             '2ndFlr', 'Bsmt', 'Pool', 'Fireplaces', 'MSZoning', 'GarageType']

df = df[variables]

df['LotArea_2'] = df['LotArea']**2
df['Age_2'] = df['Age']**2
df['GrLivArea_2'] = df['GrLivArea']**2
fig, axes = plt.subplots(1, 3, figsize = (12, 4), sharey = True)
sns.scatterplot(df, y = 'SalePrice', x = 'LotArea', ax = axes[0])
sns.scatterplot(df, y = 'SalePrice', x = 'Age', ax = axes[1])
sns.scatterplot(df, y = 'SalePrice', x = 'GrLivArea', ax = axes[2])
#plt.show()

df = pd.get_dummies(df)
#print(df.columns)





df['Age_Remodel'] = df['Remodel']*df['Age']
df['GrLivArea_2ndFlr'] = df['GrLivArea']*df['2ndFlr']
df['GrLivArea_Bsmt'] = df['GrLivArea']*df['Bsmt']

df_train = df.iloc[0:df_train.shape[0]]
df_test = df.iloc[df_train.shape[0]:]

X = df_train.drop(['SalePrice', 'Id'], axis = 1)
Y = np.log(df_train['SalePrice'])

train_valid_X, test_X, train_valid_Y, test_Y = train_test_split(X, Y, test_size = 0.2, random_state = 0)

scaler = StandardScaler()
train_valid_X = pd.DataFrame(scaler.fit_transform(train_valid_X),
                             columns = train_valid_X.columns)
test_X = pd.DataFrame(scaler.transform(test_X), columns = test_X.columns)

lr_lasso = Lasso()
param_grid = {
    'alpha': np.logspace(-6, -2, 100)
}

lr_lasso = GridSearchCV(lr_lasso, param_grid, scoring='neg_root_mean_squared_error', cv=5)
lr_lasso.fit(train_valid_X, train_valid_Y)

print("Best Lasso regularisation parameter value:", lr_lasso.best_estimator_)
print("Training score (RMSE):", - lr_lasso.score(train_valid_X, train_valid_Y))
print("Test score (RMSE):", - lr_lasso.score(test_X, test_Y))
      
lr = LinearRegression(fit_intercept = True)
lr.fit(train_valid_X, train_valid_Y)

print("Training score (coefficient of determination):", lr.score(train_valid_X, train_valid_Y))
print("Training score (RMSE):", mean_squared_error(train_valid_Y, lr.predict(train_valid_X))**0.5)
print("Test score (coefficient of determination):", lr.score(test_X, test_Y))
print("Test score (RMSE):", mean_squared_error(test_Y, lr.predict(test_X))**0.5)

lr.fit(X, Y)

predictions = np.exp(lr.predict(df_test.drop(['SalePrice', 'Id'], axis = 1)))

submission = pd.DataFrame({'Id': df_test['Id'], 'SalePrice': predictions})

submission.to_csv("submission.csv", index = False)