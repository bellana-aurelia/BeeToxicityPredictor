```python
# install libraries
!pip install catboost
!pip install xgboost
!pip install imblearn
!pip install rdkit
```

    Requirement already satisfied: catboost in /usr/local/lib/python3.11/dist-packages (1.2.8)
    Requirement already satisfied: graphviz in /usr/local/lib/python3.11/dist-packages (from catboost) (0.20.3)
    Requirement already satisfied: matplotlib in /usr/local/lib/python3.11/dist-packages (from catboost) (3.10.0)
    Requirement already satisfied: numpy<3.0,>=1.16.0 in /usr/local/lib/python3.11/dist-packages (from catboost) (2.0.2)
    Requirement already satisfied: pandas>=0.24 in /usr/local/lib/python3.11/dist-packages (from catboost) (2.2.2)
    Requirement already satisfied: scipy in /usr/local/lib/python3.11/dist-packages (from catboost) (1.15.3)
    Requirement already satisfied: plotly in /usr/local/lib/python3.11/dist-packages (from catboost) (5.24.1)
    Requirement already satisfied: six in /usr/local/lib/python3.11/dist-packages (from catboost) (1.17.0)
    Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.11/dist-packages (from pandas>=0.24->catboost) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.11/dist-packages (from pandas>=0.24->catboost) (2025.2)
    Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.11/dist-packages (from pandas>=0.24->catboost) (2025.2)
    Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.11/dist-packages (from matplotlib->catboost) (1.3.2)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.11/dist-packages (from matplotlib->catboost) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.11/dist-packages (from matplotlib->catboost) (4.58.1)
    Requirement already satisfied: kiwisolver>=1.3.1 in /usr/local/lib/python3.11/dist-packages (from matplotlib->catboost) (1.4.8)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.11/dist-packages (from matplotlib->catboost) (24.2)
    Requirement already satisfied: pillow>=8 in /usr/local/lib/python3.11/dist-packages (from matplotlib->catboost) (11.2.1)
    Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.11/dist-packages (from matplotlib->catboost) (3.2.3)
    Requirement already satisfied: tenacity>=6.2.0 in /usr/local/lib/python3.11/dist-packages (from plotly->catboost) (9.1.2)
    Requirement already satisfied: xgboost in /usr/local/lib/python3.11/dist-packages (2.1.4)
    Requirement already satisfied: numpy in /usr/local/lib/python3.11/dist-packages (from xgboost) (2.0.2)
    Requirement already satisfied: nvidia-nccl-cu12 in /usr/local/lib/python3.11/dist-packages (from xgboost) (2.21.5)
    Requirement already satisfied: scipy in /usr/local/lib/python3.11/dist-packages (from xgboost) (1.15.3)
    Requirement already satisfied: imblearn in /usr/local/lib/python3.11/dist-packages (0.0)
    Requirement already satisfied: imbalanced-learn in /usr/local/lib/python3.11/dist-packages (from imblearn) (0.13.0)
    Requirement already satisfied: numpy<3,>=1.24.3 in /usr/local/lib/python3.11/dist-packages (from imbalanced-learn->imblearn) (2.0.2)
    Requirement already satisfied: scipy<2,>=1.10.1 in /usr/local/lib/python3.11/dist-packages (from imbalanced-learn->imblearn) (1.15.3)
    Requirement already satisfied: scikit-learn<2,>=1.3.2 in /usr/local/lib/python3.11/dist-packages (from imbalanced-learn->imblearn) (1.6.1)
    Requirement already satisfied: sklearn-compat<1,>=0.1 in /usr/local/lib/python3.11/dist-packages (from imbalanced-learn->imblearn) (0.1.3)
    Requirement already satisfied: joblib<2,>=1.1.1 in /usr/local/lib/python3.11/dist-packages (from imbalanced-learn->imblearn) (1.5.1)
    Requirement already satisfied: threadpoolctl<4,>=2.0.0 in /usr/local/lib/python3.11/dist-packages (from imbalanced-learn->imblearn) (3.6.0)
    Requirement already satisfied: rdkit in /usr/local/lib/python3.11/dist-packages (2025.3.3)
    Requirement already satisfied: numpy in /usr/local/lib/python3.11/dist-packages (from rdkit) (2.0.2)
    Requirement already satisfied: Pillow in /usr/local/lib/python3.11/dist-packages (from rdkit) (11.2.1)
    


```python
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import VotingClassifier, BaggingClassifier
from sklearn.utils._param_validation import generate_valid_param, validate_parameter_constraints
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFECV
from rdkit import Chem
from rdkit.Chem import Descriptors
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
```

# Data Preprocessing


```python
# read dataset
df = pd.read_csv('apistox_dataset.csv')
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>CID</th>
      <th>CAS</th>
      <th>SMILES</th>
      <th>source</th>
      <th>year</th>
      <th>toxicity_type</th>
      <th>herbicide</th>
      <th>fungicide</th>
      <th>insecticide</th>
      <th>other_agrochemical</th>
      <th>label</th>
      <th>ppdb_level</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ethanedioic acid</td>
      <td>971</td>
      <td>144-62-7</td>
      <td>O=C(O)C(=O)O</td>
      <td>ECOTOX</td>
      <td>1832</td>
      <td>Contact</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Para-cymene</td>
      <td>7463</td>
      <td>99-87-6</td>
      <td>Cc1ccc(C(C)C)cc1</td>
      <td>BPDB</td>
      <td>1833</td>
      <td>Other</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Kieselguhr</td>
      <td>24261</td>
      <td>61790-53-2</td>
      <td>O=[Si]=O</td>
      <td>ECOTOX</td>
      <td>1833</td>
      <td>Contact</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Benzoic acid</td>
      <td>243</td>
      <td>65-85-0</td>
      <td>O=C(O)c1ccccc1</td>
      <td>ECOTOX</td>
      <td>1833</td>
      <td>Contact</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Tetradifon (Ref: ENT 23737)</td>
      <td>8305</td>
      <td>116-29-0</td>
      <td>O=S(=O)(c1ccc(Cl)cc1)c1cc(Cl)c(Cl)cc1Cl</td>
      <td>PPDB</td>
      <td>1836</td>
      <td>Oral</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1030</th>
      <td>Benzyl 4-amino-3-chloro-6-(4-chloro-2-fluoro-3...</td>
      <td>70495450</td>
      <td>1390661-72-9</td>
      <td>COc1c(Cl)ccc(-c2nc(C(=O)OCc3ccccc3)c(Cl)c(N)c2...</td>
      <td>ECOTOX</td>
      <td>2023</td>
      <td>Contact</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1031</th>
      <td>Glyphosate-dimethylammonium</td>
      <td>13208144</td>
      <td>34494-04-7</td>
      <td>CNC.O=C(O)CNCP(=O)(O)O</td>
      <td>PPDB</td>
      <td>2023</td>
      <td>Contact</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1032</th>
      <td>Glyphosate-monoammonium</td>
      <td>11615303</td>
      <td>114370-14-8</td>
      <td>N.O=C(O)CNCP(=O)(O)O</td>
      <td>PPDB</td>
      <td>2023</td>
      <td>Contact</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1033</th>
      <td>Bixlozone (Ref: F9600)</td>
      <td>15056663</td>
      <td>81777-95-9</td>
      <td>CC1(C)CON(Cc2ccc(Cl)cc2Cl)C1=O</td>
      <td>PPDB</td>
      <td>2023</td>
      <td>Contact</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1034</th>
      <td>2-(4-Chloro-2-methylphenoxy)acetic acid compd....</td>
      <td>62420</td>
      <td>2039-46-5</td>
      <td>CNC.Cc1cc(Cl)ccc1OCC(=O)O</td>
      <td>ECOTOX</td>
      <td>2023</td>
      <td>Contact</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>1035 rows × 13 columns</p>
</div>




```python
# check for null values
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1035 entries, 0 to 1034
    Data columns (total 13 columns):
     #   Column              Non-Null Count  Dtype 
    ---  ------              --------------  ----- 
     0   name                1035 non-null   object
     1   CID                 1035 non-null   int64 
     2   CAS                 1035 non-null   object
     3   SMILES              1035 non-null   object
     4   source              1035 non-null   object
     5   year                1035 non-null   int64 
     6   toxicity_type       1035 non-null   object
     7   herbicide           1035 non-null   int64 
     8   fungicide           1035 non-null   int64 
     9   insecticide         1035 non-null   int64 
     10  other_agrochemical  1035 non-null   int64 
     11  label               1035 non-null   int64 
     12  ppdb_level          1035 non-null   int64 
    dtypes: int64(8), object(5)
    memory usage: 105.2+ KB
    


```python
# drop duplicates
df = df.drop_duplicates()
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1035 entries, 0 to 1034
    Data columns (total 13 columns):
     #   Column              Non-Null Count  Dtype 
    ---  ------              --------------  ----- 
     0   name                1035 non-null   object
     1   CID                 1035 non-null   int64 
     2   CAS                 1035 non-null   object
     3   SMILES              1035 non-null   object
     4   source              1035 non-null   object
     5   year                1035 non-null   int64 
     6   toxicity_type       1035 non-null   object
     7   herbicide           1035 non-null   int64 
     8   fungicide           1035 non-null   int64 
     9   insecticide         1035 non-null   int64 
     10  other_agrochemical  1035 non-null   int64 
     11  label               1035 non-null   int64 
     12  ppdb_level          1035 non-null   int64 
    dtypes: int64(8), object(5)
    memory usage: 105.2+ KB
    

# EDA 1


```python
# color palette
bee_palette = [
    "#D18900",  # kuning terang
    "#F6C747",  # kuning madu
    "#FAD66B",  # kuning pucat
    "#EDA63A",  # orange hangat
    "#E2853B",  # oranye keemasan
    "#C97733",  # caramel
    "#A35A36",  # coklat kayu
    "#6A4D3B",  # coklat tua
    "#4E3C34",  # hitam madu
    "#BF5E3B",  # merah bata
]
```


```python
# Choose colors for label
custom_palette = {
    'Non-Toxic': bee_palette[1],  # "#D18900"
    'Toxic': bee_palette[9]       # "#BF5E3B"
}

# Counting number of Non-Toxic and Toxic based on "source"
data = []
sources = df['source'].unique()

for src in sources:
    non_toxic = df[(df['label'] == 0) & (df['source'] == src)].shape[0]
    toxic = df[(df['label'] == 1) & (df['source'] == src)].shape[0]
    data.append({'Source': src, 'Label': 'Non-Toxic', 'Count': non_toxic})
    data.append({'Source': src, 'Label': 'Toxic', 'Count': toxic})

plot_df = pd.DataFrame(data)

# Make Plot
plt.figure(figsize=(12, 6))
sns.barplot(data=plot_df, x='Source', y='Count', hue='Label', palette=custom_palette)
plt.title('Toxic vs Non-Toxic Compounds by Source Category')
plt.xlabel('Source')
plt.ylabel('Number of Compounds')

# Adding number to bar
for p in plt.gca().patches:
    height = p.get_height()
    if height > 0:
        plt.gca().annotate(f'{int(height)}',
                           (p.get_x() + p.get_width() / 2., height),
                           ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()
```


    
![png](output_8_0.png)
    



```python
# Choosing colors for label
custom_palette = {
    'Non-Toxic': bee_palette[1],  # "#D18900"
    'Toxic': bee_palette[9]       # "#BF5E3B"
}

# Prepare data
categories = ['herbicide', 'insecticide', 'fungicide', 'other_agrochemical']
data = []

for cat in categories:
    non_toxic = df[df['label'] == 0][cat].sum()
    toxic = df[df['label'] == 1][cat].sum()
    data.append({'Category': cat.capitalize(), 'Label': 'Non-Toxic', 'Count': non_toxic})
    data.append({'Category': cat.capitalize(), 'Label': 'Toxic', 'Count': toxic})

plot_df = pd.DataFrame(data)

# Make Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=plot_df, x='Category', y='Count', hue='Label', palette=custom_palette)
plt.title('Toxic vs Non-Toxic Compounds per Agrochemical Category')
plt.xlabel('Agrochemical Category')
plt.ylabel('Number of Compounds')

# Adding number to bar
for p in plt.gca().patches:
    height = p.get_height()
    if height > 0:
        plt.gca().annotate(f'{int(height)}',
                           (p.get_x() + p.get_width() / 2., height),
                           ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()
```


    
![png](output_9_0.png)
    



```python
# Set color
hist_color = "#D18900"
box_color = "#FAD66B"

# Get numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Make plot
for col in numerical_cols:
    try:
        plt.figure(figsize=(12, 5))

        # histogram
        plt.subplot(1, 2, 1)
        sns.histplot(df[col].dropna(), kde=True, color=hist_color, bins=30)
        plt.title(f'Distribution of "{col}"')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.grid(True, linestyle='--', linewidth=0.5)

        # boxplot & outliers
        plt.subplot(1, 2, 2)
        sns.boxplot(y=df[col].dropna(), color=box_color)
        plt.title(f'Outliers & Value Spread: "{col}"')
        plt.ylabel(col)
        plt.grid(True, axis='y', linestyle='--', linewidth=0.5)

        plt.suptitle(f'Distribution and Outlier Analysis for "{col}"', fontsize=14, y=1.05)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    except Exception as e:
        print(f"Skipped column '{col}' due to error: {e}")

```


    
![png](output_10_0.png)
    



    
![png](output_10_1.png)
    



    
![png](output_10_2.png)
    



    
![png](output_10_3.png)
    



    
![png](output_10_4.png)
    



    
![png](output_10_5.png)
    



    
![png](output_10_6.png)
    



    
![png](output_10_7.png)
    



```python
# Get categorical columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns

# Make pie chart
for col in categorical_cols:
    try:
        # Count distributions for label: toxic (1)
        toxic_counts = df[df['label'] == 1].groupby(col)['label'].count().sort_values(ascending=False)
        if len(toxic_counts) > 10:
            toxic_counts = toxic_counts.head(10)

        # Count distributions for label: non-toxic (0)
        nontoxic_counts = df[df['label'] == 0].groupby(col)['label'].count().sort_values(ascending=False)
        if len(nontoxic_counts) > 10:
            nontoxic_counts = nontoxic_counts.head(10)

        # --- PIE CHART FOR TOXIC COMPOUNDS ---
        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        plt.pie(
            x=toxic_counts.values,
            labels=toxic_counts.index,
            autopct='%1.1f%%',
            colors=bee_palette[:len(toxic_counts)],
            startangle=140,
            wedgeprops={'edgecolor': 'white'}
        )
        plt.title(f"Toxic Compounds in '{col}'", fontsize=11)

        # --- PIE CHART FOR NON-TOXIC COMPOUNDS ---
        plt.subplot(1, 2, 2)
        plt.pie(
            x=nontoxic_counts.values,
            labels=nontoxic_counts.index,
            autopct='%1.1f%%',
            colors=bee_palette[:len(nontoxic_counts)],
            startangle=140,
            wedgeprops={'edgecolor': 'white'}
        )
        plt.title(f"Non-Toxic Compounds in '{col}'", fontsize=11)

        plt.suptitle(f"Compound Distribution Across '{col}' Categories by Toxicity", fontsize=13)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    except Exception as e:
        print(f"Skipped column '{col}' due to error: {e}")

```

    C:\Users\aaron\AppData\Local\Temp\ipykernel_4576\3733019961.py:44: UserWarning: Tight layout not applied. tight_layout cannot make axes width small enough to accommodate all axes decorations
      plt.tight_layout(rect=[0, 0, 1, 0.95])
    


    
![png](output_11_1.png)
    



    
![png](output_11_2.png)
    



    
![png](output_11_3.png)
    



    
![png](output_11_4.png)
    



    
![png](output_11_5.png)
    


# Feature Engineering

Feature engineering is selection, creation or transformation of features which could help improve the performance of machine learning models. In our dataset, there is one feature that quite difficult to be interpreted by models. That being the "SMILES" feature. 

SMILES is the abbreviation of Simplified Molecular Input Line Entry System which is a chemical notation that allows a user to represent a chemical structure in a way that can be used by the computer. The SMILES feature is processed using a library called RDKit which is designed to handle chemistry related stuff. Using this library, we are able to convert our SMILES feature into 5 new features that the models would easily understand. 

The features chosen are possible characteristics that allow us to identify whether a molecule is toxic or not. These new features include: 'MolWt' or molecular weight, 'TPSA' or topological polar surface area, 'NumHDonors' or number of hydrogen bond donors, 'NumHAcceptors' or number of hydrogen bond acceptors and 'MolLogP' or octanol water partition coefficient.


```python
# turn SMILES column to features
def smiles_to_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    return {
        'MolWt': Descriptors.MolWt(mol),
        'TPSA': Descriptors.TPSA(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'MolLogP': Descriptors.MolLogP(mol),
    }

features = df['SMILES'].apply(smiles_to_features)
features_df = pd.DataFrame(features.tolist())
df = pd.concat([df, features_df], axis=1)
df = df.drop('SMILES', axis=1)
```

Other than creating new features, we transform categorical features by using one hot encoding. In our dataset, there is only one categorical feature that hasn't been transformed which is "toxicity_type". There are three classes in "toxicity_type": contact, oral and other. New columns for each class is created and in those columns are True or False values depending what their original class were. We also set drop_first to True which drops contact. This is to reduce the load that the model need to process which could increase its performance.


```python
# encode categorical values
df = pd.get_dummies(df, columns=['toxicity_type'], drop_first=True)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>CID</th>
      <th>CAS</th>
      <th>source</th>
      <th>year</th>
      <th>herbicide</th>
      <th>fungicide</th>
      <th>insecticide</th>
      <th>other_agrochemical</th>
      <th>label</th>
      <th>ppdb_level</th>
      <th>MolWt</th>
      <th>TPSA</th>
      <th>NumHDonors</th>
      <th>NumHAcceptors</th>
      <th>MolLogP</th>
      <th>toxicity_type_Oral</th>
      <th>toxicity_type_Other</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ethanedioic acid</td>
      <td>971</td>
      <td>144-62-7</td>
      <td>ECOTOX</td>
      <td>1832</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>90.034</td>
      <td>74.60</td>
      <td>2</td>
      <td>2</td>
      <td>-0.84440</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Para-cymene</td>
      <td>7463</td>
      <td>99-87-6</td>
      <td>BPDB</td>
      <td>1833</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>134.222</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>3.11842</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Kieselguhr</td>
      <td>24261</td>
      <td>61790-53-2</td>
      <td>ECOTOX</td>
      <td>1833</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>60.084</td>
      <td>34.14</td>
      <td>0</td>
      <td>2</td>
      <td>-0.61840</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Benzoic acid</td>
      <td>243</td>
      <td>65-85-0</td>
      <td>ECOTOX</td>
      <td>1833</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>122.123</td>
      <td>37.30</td>
      <td>1</td>
      <td>1</td>
      <td>1.38480</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Tetradifon (Ref: ENT 23737)</td>
      <td>8305</td>
      <td>116-29-0</td>
      <td>PPDB</td>
      <td>1836</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>356.057</td>
      <td>34.14</td>
      <td>0</td>
      <td>2</td>
      <td>5.13300</td>
      <td>True</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1035 entries, 0 to 1034
    Data columns (total 18 columns):
     #   Column               Non-Null Count  Dtype  
    ---  ------               --------------  -----  
     0   name                 1035 non-null   object 
     1   CID                  1035 non-null   int64  
     2   CAS                  1035 non-null   object 
     3   source               1035 non-null   object 
     4   year                 1035 non-null   int64  
     5   herbicide            1035 non-null   int64  
     6   fungicide            1035 non-null   int64  
     7   insecticide          1035 non-null   int64  
     8   other_agrochemical   1035 non-null   int64  
     9   label                1035 non-null   int64  
     10  ppdb_level           1035 non-null   int64  
     11  MolWt                1035 non-null   float64
     12  TPSA                 1035 non-null   float64
     13  NumHDonors           1035 non-null   int64  
     14  NumHAcceptors        1035 non-null   int64  
     15  MolLogP              1035 non-null   float64
     16  toxicity_type_Oral   1035 non-null   bool   
     17  toxicity_type_Other  1035 non-null   bool   
    dtypes: bool(2), float64(3), int64(10), object(3)
    memory usage: 131.5+ KB
    

# EDA 2 (after feature engineering)


```python
# Show imbalance in label
bee_colors = ['#FAD66B', '#A35A36']  # Non-toxic, Toxic


sns.countplot(data=df, x="label", palette=bee_colors)

plt.title("Toxic vs. Non-toxic")
plt.xticks([0, 1], ['Non-toxic', 'Toxic'])
plt.ylabel("Count")
plt.xlabel("Category (Label 0 = Non-Toxic, 1 = Toxic)")
plt.tight_layout()
plt.show()
```


    
![png](output_19_0.png)
    



```python
from matplotlib.colors import LinearSegmentedColormap

# Set colors
bee_strong_contrast = [
    "#6B3E26",
    "#B36B3E",
    "#FFF9D1",
    "#F6B93B",
    "#A65C00",
]

# Make heatmap
bee_cmap_strong = LinearSegmentedColormap.from_list("bee_strong_contrast", bee_strong_contrast, N=256)

num_cols = df.select_dtypes(include='number')

corr = num_cols.corr(numeric_only=True)

plt.figure(figsize=(12, 10))
sns.heatmap(
    corr,
    annot=True,
    fmt='.2f',
    cmap=bee_cmap_strong,
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.75}
)
plt.title('Heatmap of Numerical Feature Correlations', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.show()
```


    
![png](output_20_0.png)
    



```python
import math

# Get newly added features
numerical_cols = ['MolWt', 'TPSA', 'NumHDonors', 'NumHAcceptors', 'MolLogP']

n_cols = 3
n_rows = math.ceil(len(numerical_cols) / n_cols)

# Make plot
for i, col in enumerate(numerical_cols):
    plt.figure(figsize=(12, 5))

    # histogram
    plt.subplot(1, 2, 1)
    sns.histplot(df[col].dropna(), kde=True, color=hist_color, bins=30)
    plt.title(f'Distribution of "{col}"')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.grid(True, linestyle='--', linewidth=0.5)

    # boxplot & outliers
    plt.subplot(1, 2, 2)
    sns.boxplot(y=df[col].dropna(), color=box_color)
    plt.title(f'Outliers & Value Spread: "{col}"')
    plt.ylabel(col)
    plt.grid(True, axis='y', linestyle='--', linewidth=0.5)

    plt.suptitle(f'Distribution and Outlier Analysis for "{col}"', fontsize=14, y=1.05)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

plt.tight_layout()
plt.show()
```


    
![png](output_21_0.png)
    



    
![png](output_21_1.png)
    



    
![png](output_21_2.png)
    



    
![png](output_21_3.png)
    



    
![png](output_21_4.png)
    



    <Figure size 640x480 with 0 Axes>


After performing feature engineering, we visualized the new numerical features (MolWt, TPSA, MolLogP, NumDonors, NumHAcceptors) as seen in the graphs above. However, these outliers are not considered problematic. In chemical datasets, such outliers often represent unique molecular structures, in this case, it can be highly informative for toxicity predictions.  

# MODELLING + EVALUATION

This section includes training and evaluating the model. Evaluation is done using a confusion matrix which records the number of True Positives, True Negatives, False Positives and False Negatives from the prediction and actual label. In a confusion matrix, multiple evaluation metrics can be gained. For example, the most common metrics used are accuracy and f1-score. For this evaluation, we are using lowest f1-score as the target variable 'label' is imbalanced. These metrics are gained using the function classification_report from sklearn.metrics library. A ROC-Curve is also made to evaluate the general performance of each model.

Hyperparameter tuning is also done using grid search for all the models. The evaluation of the best model as well as their best parameters are stated below.


```python
# dropping unused columns
df = df.drop(['name', 'CID', 'CAS', 'source', 'year', 'ppdb_level'], axis=1)
```


```python
# split data to train and test
X = df.drop(columns=['label'], axis = 1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 21)

# scaling x
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### RANDOM FOREST


```python
rf = RandomForestClassifier(random_state = 21)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
```

    Confusion Matrix:
    [[200  22]
     [ 31  58]]
                  precision    recall  f1-score   support
    
               0       0.87      0.90      0.88       222
               1       0.72      0.65      0.69        89
    
        accuracy                           0.83       311
       macro avg       0.80      0.78      0.78       311
    weighted avg       0.83      0.83      0.83       311
    
    


```python
importances = rf.feature_importances_
feature_names = X.columns
sorted_idx = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 5))
plt.barh(range(len(importances)), importances[sorted_idx], color= "#EDA63A")
plt.yticks(range(len(importances)), [feature_names[i] for i in sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Random Forest - Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

```


    
![png](output_29_0.png)
    



```python
param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 4, 5],
    'max_features': ['sqrt', 'log2'],
    'random_state': [21]
}

# GridSearchCV with scoring based on F1
grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Fit on training data
grid.fit(X_train, y_train)

# Best model
best_rf = grid.best_estimator_
print("Best parameters:", grid.best_params_)

# Evaluate
y_pred = best_rf.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))
```

    Fitting 5 folds for each of 36 candidates, totalling 180 fits
    Best parameters: {'max_depth': 5, 'max_features': 'sqrt', 'min_samples_split': 5, 'n_estimators': 100, 'random_state': 21}
    
    Confusion Matrix:
    [[207  15]
     [ 39  50]]
    
    Classification Report
                  precision    recall  f1-score   support
    
               0       0.84      0.93      0.88       222
               1       0.77      0.56      0.65        89
    
        accuracy                           0.83       311
       macro avg       0.81      0.75      0.77       311
    weighted avg       0.82      0.83      0.82       311
    
    

### LOGISTIC REGRESSION


```python
lr = LogisticRegression(random_state = 21)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

    Confusion Matrix:
    [[200  22]
     [ 35  54]]
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.85      0.90      0.88       222
               1       0.71      0.61      0.65        89
    
        accuracy                           0.82       311
       macro avg       0.78      0.75      0.76       311
    weighted avg       0.81      0.82      0.81       311
    
    


```python
param_grid = [
    {'penalty':['l1','l2','none'],
    'C' : [0.01, 0.1, 1],
    'solver': ['lbfgs','newton-cg'],
    'max_iter'  : [500, 300],
    'random_state': [21]
}
]

grid = GridSearchCV(
    estimator=lr,
    param_grid=param_grid,
    scoring='f1'
)

# Fit on training data
grid.fit(X_train, y_train)

# Best model
best_lr = grid.best_estimator_
print("Best parameters:", grid.best_params_)

# Evaluate
y_pred = best_lr.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))
# Scroll all the way down to see report
```

    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1181: UserWarning: Setting penalty=None will ignore the C and l1_ratio parameters
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1181: UserWarning: Setting penalty=None will ignore the C and l1_ratio parameters
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1181: UserWarning: Setting penalty=None will ignore the C and l1_ratio parameters
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1181: UserWarning: Setting penalty=None will ignore the C and l1_ratio parameters
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1181: UserWarning: Setting penalty=None will ignore the C and l1_ratio parameters
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1181: UserWarning: Setting penalty=None will ignore the C and l1_ratio parameters
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1181: UserWarning: Setting penalty=None will ignore the C and l1_ratio parameters
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1181: UserWarning: Setting penalty=None will ignore the C and l1_ratio parameters
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1181: UserWarning: Setting penalty=None will ignore the C and l1_ratio parameters
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1181: UserWarning: Setting penalty=None will ignore the C and l1_ratio parameters
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1181: UserWarning: Setting penalty=None will ignore the C and l1_ratio parameters
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1181: UserWarning: Setting penalty=None will ignore the C and l1_ratio parameters
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1181: UserWarning: Setting penalty=None will ignore the C and l1_ratio parameters
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1181: UserWarning: Setting penalty=None will ignore the C and l1_ratio parameters
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1181: UserWarning: Setting penalty=None will ignore the C and l1_ratio parameters
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1181: UserWarning: Setting penalty=None will ignore the C and l1_ratio parameters
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1181: UserWarning: Setting penalty=None will ignore the C and l1_ratio parameters
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1181: UserWarning: Setting penalty=None will ignore the C and l1_ratio parameters
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1181: UserWarning: Setting penalty=None will ignore the C and l1_ratio parameters
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1181: UserWarning: Setting penalty=None will ignore the C and l1_ratio parameters
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1181: UserWarning: Setting penalty=None will ignore the C and l1_ratio parameters
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1181: UserWarning: Setting penalty=None will ignore the C and l1_ratio parameters
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1181: UserWarning: Setting penalty=None will ignore the C and l1_ratio parameters
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1181: UserWarning: Setting penalty=None will ignore the C and l1_ratio parameters
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1181: UserWarning: Setting penalty=None will ignore the C and l1_ratio parameters
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1181: UserWarning: Setting penalty=None will ignore the C and l1_ratio parameters
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1181: UserWarning: Setting penalty=None will ignore the C and l1_ratio parameters
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1181: UserWarning: Setting penalty=None will ignore the C and l1_ratio parameters
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1181: UserWarning: Setting penalty=None will ignore the C and l1_ratio parameters
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1181: UserWarning: Setting penalty=None will ignore the C and l1_ratio parameters
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1181: UserWarning: Setting penalty=None will ignore the C and l1_ratio parameters
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1181: UserWarning: Setting penalty=None will ignore the C and l1_ratio parameters
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1181: UserWarning: Setting penalty=None will ignore the C and l1_ratio parameters
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1181: UserWarning: Setting penalty=None will ignore the C and l1_ratio parameters
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1181: UserWarning: Setting penalty=None will ignore the C and l1_ratio parameters
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1181: UserWarning: Setting penalty=None will ignore the C and l1_ratio parameters
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1181: UserWarning: Setting penalty=None will ignore the C and l1_ratio parameters
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1181: UserWarning: Setting penalty=None will ignore the C and l1_ratio parameters
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1181: UserWarning: Setting penalty=None will ignore the C and l1_ratio parameters
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1181: UserWarning: Setting penalty=None will ignore the C and l1_ratio parameters
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\model_selection\_validation.py:378: FitFailedWarning: 
    60 fits failed out of a total of 180.
    The score on these train-test partitions for these parameters will be set to nan.
    If these failures are not expected, you can try to debug them by setting error_score='raise'.
    
    Below are more details about the failures:
    --------------------------------------------------------------------------------
    30 fits failed with the following error:
    Traceback (most recent call last):
      File "C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\model_selection\_validation.py", line 686, in _fit_and_score
        estimator.fit(X_train, y_train, **fit_params)
      File "C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py", line 1162, in fit
        solver = _check_solver(self.solver, self.penalty, self.dual)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py", line 54, in _check_solver
        raise ValueError(
    ValueError: Solver lbfgs supports only 'l2' or 'none' penalties, got l1 penalty.
    
    --------------------------------------------------------------------------------
    30 fits failed with the following error:
    Traceback (most recent call last):
      File "C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\model_selection\_validation.py", line 686, in _fit_and_score
        estimator.fit(X_train, y_train, **fit_params)
      File "C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py", line 1162, in fit
        solver = _check_solver(self.solver, self.penalty, self.dual)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py", line 54, in _check_solver
        raise ValueError(
    ValueError: Solver newton-cg supports only 'l2' or 'none' penalties, got l1 penalty.
    
      warnings.warn(some_fits_failed_message, FitFailedWarning)
    

    Best parameters: {'C': 0.01, 'max_iter': 500, 'penalty': 'none', 'random_state': 21, 'solver': 'lbfgs'}
    
    Confusion Matrix:
    [[200  22]
     [ 36  53]]
    
    Classification Report
                  precision    recall  f1-score   support
    
               0       0.85      0.90      0.87       222
               1       0.71      0.60      0.65        89
    
        accuracy                           0.81       311
       macro avg       0.78      0.75      0.76       311
    weighted avg       0.81      0.81      0.81       311
    
    

    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\model_selection\_search.py:952: UserWarning: One or more of the test scores are non-finite: [       nan        nan 0.6318259  0.6318259  0.66630072 0.66630072
            nan        nan 0.6318259  0.6318259  0.66630072 0.66630072
            nan        nan 0.64675428 0.64675428 0.66630072 0.66630072
            nan        nan 0.64675428 0.64675428 0.66630072 0.66630072
            nan        nan 0.66284871 0.66284871 0.66630072 0.66630072
            nan        nan 0.66284871 0.66284871 0.66630072 0.66630072]
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1181: UserWarning: Setting penalty=None will ignore the C and l1_ratio parameters
      warnings.warn(
    

### DECISION TREE


```python
tree = DecisionTreeClassifier(random_state = 21)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

    Confusion Matrix:
    [[183  39]
     [ 41  48]]
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.82      0.82      0.82       222
               1       0.55      0.54      0.55        89
    
        accuracy                           0.74       311
       macro avg       0.68      0.68      0.68       311
    weighted avg       0.74      0.74      0.74       311
    
    


```python
param_grid = {
    'max_depth': [10, 20, 5],
    'min_samples_split': [2, 4, 5],
    'min_samples_leaf': [4, 2, 3],
    'random_state' : [21]
}

# GridSearchCV with scoring based on F1
grid = GridSearchCV(
    estimator=tree,
    param_grid=param_grid,
    scoring='f1',
    verbose=1
)

# Fit on training data
grid.fit(X_train, y_train)

# Best model
best_tree = grid.best_estimator_
print("Best parameters:", grid.best_params_)

# Evaluate
y_pred = best_tree.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))
```

    Fitting 5 folds for each of 27 candidates, totalling 135 fits
    Best parameters: {'max_depth': 5, 'min_samples_leaf': 2, 'min_samples_split': 5, 'random_state': 21}
    
    Confusion Matrix:
    [[198  24]
     [ 40  49]]
    
    Classification Report
                  precision    recall  f1-score   support
    
               0       0.83      0.89      0.86       222
               1       0.67      0.55      0.60        89
    
        accuracy                           0.79       311
       macro avg       0.75      0.72      0.73       311
    weighted avg       0.79      0.79      0.79       311
    
    

### SVM


```python
svm = SVC(probability=True, random_state = 21)

svm.fit(X_train,y_train)

y_pred = svm.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

    Confusion Matrix:
    [[198  24]
     [ 34  55]]
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.85      0.89      0.87       222
               1       0.70      0.62      0.65        89
    
        accuracy                           0.81       311
       macro avg       0.77      0.75      0.76       311
    weighted avg       0.81      0.81      0.81       311
    
    


```python
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto'],
    'random_state' : [21]
}

grid = GridSearchCV(
    estimator=svm,
    param_grid=param_grid,
    scoring='f1',
    verbose=1
)

# Fit on training data
grid.fit(X_train, y_train)

# Best model
best_svm = grid.best_estimator_
print("Best parameters:", grid.best_params_)

# Evaluate
y_pred = best_svm.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))
```

    Fitting 5 folds for each of 16 candidates, totalling 80 fits
    Best parameters: {'C': 1, 'gamma': 'scale', 'kernel': 'linear', 'random_state': 21}
    
    Confusion Matrix:
    [[195  27]
     [ 33  56]]
    
    Classification Report
                  precision    recall  f1-score   support
    
               0       0.86      0.88      0.87       222
               1       0.67      0.63      0.65        89
    
        accuracy                           0.81       311
       macro avg       0.76      0.75      0.76       311
    weighted avg       0.80      0.81      0.80       311
    
    

### K-NEAREST NEIGHBOR


```python
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

    Confusion Matrix:
    [[199  23]
     [ 32  57]]
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.86      0.90      0.88       222
               1       0.71      0.64      0.67        89
    
        accuracy                           0.82       311
       macro avg       0.79      0.77      0.78       311
    weighted avg       0.82      0.82      0.82       311
    
    


```python
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

grid = GridSearchCV(
    estimator=knn,
    param_grid=param_grid,
    scoring='f1',
    verbose=1
)

# Fit on training data
grid.fit(X_train, y_train)

# Best model
best_knn = grid.best_estimator_
print("Best parameters:", grid.best_params_)

# Evaluate
y_pred = best_knn.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))
```

    Fitting 5 folds for each of 20 candidates, totalling 100 fits
    Best parameters: {'metric': 'euclidean', 'n_neighbors': 9, 'weights': 'distance'}
    
    Confusion Matrix:
    [[199  23]
     [ 29  60]]
    
    Classification Report
                  precision    recall  f1-score   support
    
               0       0.87      0.90      0.88       222
               1       0.72      0.67      0.70        89
    
        accuracy                           0.83       311
       macro avg       0.80      0.79      0.79       311
    weighted avg       0.83      0.83      0.83       311
    
    

### XG BOOST


```python
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state = 21)
xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

    Confusion Matrix:
    [[197  25]
     [ 34  55]]
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.85      0.89      0.87       222
               1       0.69      0.62      0.65        89
    
        accuracy                           0.81       311
       macro avg       0.77      0.75      0.76       311
    weighted avg       0.81      0.81      0.81       311
    
    

    C:\Users\aaron\anaconda3\Lib\site-packages\xgboost\training.py:183: UserWarning: [23:05:29] WARNING: C:\actions-runner\_work\xgboost\xgboost\src\learner.cc:738: 
    Parameters: { "use_label_encoder" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    

### BAGGING


```python
bagging = BaggingClassifier(estimator=best_knn, random_state = 21)

bagging.fit(X_train, y_train)
y_pred = bagging.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

    Confusion Matrix:
    [[200  22]
     [ 31  58]]
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.87      0.90      0.88       222
               1       0.72      0.65      0.69        89
    
        accuracy                           0.83       311
       macro avg       0.80      0.78      0.78       311
    weighted avg       0.83      0.83      0.83       311
    
    

### SOFT VOTING


```python
# soft voting
estimators = [('LR', best_lr),
              ('KNN', best_knn),
              ('RF', best_rf),
              ('SVM', best_svm),
              ('XGB', xgb),
              ]

voting_model = VotingClassifier(estimators = estimators, voting='soft')

voting_model.fit(X_train, y_train)

pred = voting_model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))

print("\nClassification Report:")
print(classification_report(y_test, pred))
```

    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1181: UserWarning: Setting penalty=None will ignore the C and l1_ratio parameters
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\xgboost\training.py:183: UserWarning: [23:05:34] WARNING: C:\actions-runner\_work\xgboost\xgboost\src\learner.cc:738: 
    Parameters: { "use_label_encoder" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    

    Confusion Matrix:
    [[206  16]
     [ 35  54]]
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.85      0.93      0.89       222
               1       0.77      0.61      0.68        89
    
        accuracy                           0.84       311
       macro avg       0.81      0.77      0.78       311
    weighted avg       0.83      0.84      0.83       311
    
    

### HARD VOTING


```python
# hard voting
estimators = [('LR', best_lr),
              ('KNN', best_knn),
              ('RF', best_rf),
              ('SVM', best_svm),
              ('XGB', xgb),
              ]

voting_model = VotingClassifier(estimators = estimators, voting='hard')

voting_model.fit(X_train, y_train)

pred = voting_model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))

print("\nClassification Report:")
print(classification_report(y_test, pred))
```

    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1173: FutureWarning: `penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`.
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:1181: UserWarning: Setting penalty=None will ignore the C and l1_ratio parameters
      warnings.warn(
    C:\Users\aaron\anaconda3\Lib\site-packages\xgboost\training.py:183: UserWarning: [23:05:35] WARNING: C:\actions-runner\_work\xgboost\xgboost\src\learner.cc:738: 
    Parameters: { "use_label_encoder" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    

    Confusion Matrix:
    [[202  20]
     [ 34  55]]
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.86      0.91      0.88       222
               1       0.73      0.62      0.67        89
    
        accuracy                           0.83       311
       macro avg       0.79      0.76      0.78       311
    weighted avg       0.82      0.83      0.82       311
    
    

### ADABOOST


```python
ada = AdaBoostClassifier(estimator = best_rf, random_state = 21)

ada.fit(X_train, y_train)
y_pred = ada.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

    Confusion Matrix:
    [[197  25]
     [ 31  58]]
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.86      0.89      0.88       222
               1       0.70      0.65      0.67        89
    
        accuracy                           0.82       311
       macro avg       0.78      0.77      0.77       311
    weighted avg       0.82      0.82      0.82       311
    
    

# SUMMARY MODELS

This section contains the summary of scores for all the models. The ROC Curve for all models is also here.


```python
# before tuning

# Get scores
models = ['Logistic Regression', 'Random Forest', 'Decision Tree', 'SVM', 'KNN']
f1_scores = [0.65, 0.69, 0.55, 0.65, 0.67]

# Plot scores
plt.figure(figsize=(8, 5))
sns.set(style="whitegrid")
barplot = sns.barplot(x=f1_scores, y=models, palette="viridis")

for i, score in enumerate(f1_scores):
    plt.text(score + 0.01, i, f"{score:.2f}", color='black', va='center')

plt.xlabel("F1 Score")
plt.title("F1 Scores of Models Before Tuning")
plt.xlim(0, 1)
plt.tight_layout()
plt.show()
```


    
![png](output_55_0.png)
    


Before tuning, we could see that random forest has the highest f1-score with 0.69. On the other hand, decision tree has the lowest f1-score with 0.55.


```python
# after tuning

# Get scores
models = ['Logistic Regression', 'Random Forest', 'Decision Tree', 'SVM', 'KNN']
f1_scores = [0.65, 0.65, 0.60, 0.65, 0.70]

# Plot scores
plt.figure(figsize=(8, 5))
sns.set(style="whitegrid")
barplot = sns.barplot(x=f1_scores, y=models, palette="viridis")

for i, score in enumerate(f1_scores):
    plt.text(score + 0.01, i, f"{score:.2f}", color='black', va='center')

plt.xlabel("F1 Score")
plt.title("F1 Scores of Models After Tuning")
plt.xlim(0, 1)
plt.tight_layout()
plt.show()
```


    
![png](output_57_0.png)
    


After tuning, we could see that the f1-score for random forest dropped to 0.65. The highest f1-score this time is from KNN with 0.70. The f1-score for decision tree rises to 0.69, however, it is still the lowest f1-score.


```python
# ensemble

# Get scores
models = ['Bagging (KNN)', 'Soft Voting', 'Hard Voting', 'AdaBoost (RF)', 'XGBoost']
f1_scores = [0.69, 0.68, 0.67, 0.67, 0.65]

# Plot scores
plt.figure(figsize=(8, 5))
sns.set(style="whitegrid")
barplot = sns.barplot(x=f1_scores, y=models, palette="viridis")

for i, score in enumerate(f1_scores):
    plt.text(score + 0.01, i, f"{score:.2f}", color='black', va='center')

plt.xlabel("F1 Score")
plt.title("F1 Scores of Ensemble Models")
plt.xlim(0, 1)
plt.tight_layout()
plt.show()
```


    
![png](output_59_0.png)
    


For ensemble models, the highest f1-score is bagging using KNN with 0.69. On the other hand, the lowest f1-score is XGBoost with 0.65.


```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Get best models
models = {
    'Logistic Regression': best_lr,
    'Random Forest': best_rf,
    'Decision Tree': best_tree,
    'SVM': best_svm,
    'KNN': best_knn
}

plt.figure(figsize=(8, 6))

# Plot ROC Curve
for name, model in models.items():
    if hasattr(model, "predict_proba"):
        y_probs = model.predict_proba(X_test)[:, 1]
    else:
        y_probs = model.decision_function(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')


plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Models')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

```


    
![png](output_61_0.png)
    


From the ROC curves, we can see that random forest has the highest overall performance with 0.84 while SVM has the lowest overall performance with 0.77.

From the evaluations above we can conclude that KNN is able to identify whether a chemical is toxic despite class imbalance better than other models. However, in terms of overall performance, random forest has the highest AUC score than the other models.


```python

```
