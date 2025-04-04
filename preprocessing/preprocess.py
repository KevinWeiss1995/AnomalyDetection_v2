import dask.dataframe as dd
import numpy as np
import warnings
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
import os
import sys
import pickle
import dask
import pandas as pd
sys.path.append('..')
from utils.git import get_git_repo_root
from sklearn.model_selection import train_test_split

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)




'''
This code is set up to break down the preprocessing into stages.

Stage 1: Initial loading and missing values
Stage 2: Handle duplicates and feature drops
Stage 3: Categorical encoding
Stage 4: Undersampling
Stage 5: Scaling
Stage 6: Feature importance selection

Each stage is saved to a separate parquet file in the tmp directory, so if 
you need to stop and resume, you can start from the last completed stage. 

The code is set up to be run in a single process, but it could be parallelized
if you wanted.

The final output is saved to the data/network directory, and includes the files:
train_data.csv
test_data.csv
train_labels.csv
test_labels.csv

The train_data.csv and test_data.csv files are the features for the training and
test sets, respectively.
'''



def setup_paths():
    base_repo = get_git_repo_root()
    paths = {
        'input': os.path.join(base_repo, 'data/network/NF-CICIDS2018-v3.csv'),
        'tmp': os.path.join(base_repo, 'data/tmp'),
        'output': os.path.join(base_repo, 'data/network'),
        'transformer': os.path.join(base_repo, 'results/transformers')
    }

    for key, path in paths.items():
        if key != 'input':
            os.makedirs(path, exist_ok=True)
        
    return paths

def stage1_initial_processing(paths):
    """Load and handle missing values"""
    stage1_path = os.path.join(paths['tmp'], 'stage1.parquet')
    
    if os.path.exists(stage1_path):
        print("Stage 1 already completed, skipping...")
        return
    
    print("Stage 1: Initial loading and missing values...")
    df = dd.read_csv(paths['input'])
    
    df = df.drop('Attack', axis=1)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df.map_partitions(lambda df: df.assign(
        **{col: df[col].fillna(df[col].mean()) for col in numeric_cols}
    ))
    
    df.to_parquet(stage1_path)
    print("\nStage 1 data head:")
    print(df.head())
    print("Stage 1: Complete\n")

def stage2_preprocessing(paths):
    """Handle duplicates and feature drops"""
    stage1_path = os.path.join(paths['tmp'], 'stage1.parquet')
    stage2_path = os.path.join(paths['tmp'], 'stage2.parquet')
    
    if os.path.exists(stage2_path):
        print("Stage 2 already completed, skipping...")
        return
    
    print("Stage 2: Loading stage1 data...")
    df = dd.read_parquet(stage1_path)
    
    print("Basic preprocessing...")
    df = df.drop_duplicates()
    df = df.drop(columns=['IPV4_SRC_ADDR', 'IPV4_DST_ADDR'])
    
    df.to_parquet(stage2_path)
    print("\nStage 2 data head:")
    print(df.head())
    print("Stage 2: Complete\n")

def stage3_encoding(paths):
    """Handle categorical encoding"""
    stage2_path = os.path.join(paths['tmp'], 'stage2.parquet')
    stage3_path = os.path.join(paths['tmp'], 'stage3.parquet')
    
    if os.path.exists(stage3_path):
        print("Stage 3 already completed, skipping...")
        return
    
    print("Stage 3: Loading data from stage 2...")
    df = dd.read_parquet(stage2_path)
    
    print("Encoding categorical data...")

    df['PROTOCOL'] = df['PROTOCOL'].astype('category')
    df = df.categorize(columns=['PROTOCOL'])
    df['PROTOCOL'] = df['PROTOCOL'].cat.codes
    
    print("Saving stage 3 output...")
    df.to_parquet(stage3_path)
    print("\nStage 3 data head:")
    print(df.head())
    print("Stage 3: Complete\n")

def stage4_undersampling(paths):
    """Handle class imbalance"""
    stage3_path = os.path.join(paths['tmp'], 'stage3.parquet')
    stage4_path = os.path.join(paths['tmp'], 'stage4.parquet')
    
    if os.path.exists(stage4_path):
        print("Stage 4 output exists, skipping to next stage...")
        return
    
    print("Stage 4: Loading data from stage 3...")
    df = dd.read_parquet(stage3_path)
    
    print("Undersampling...")
    print("Converting to pandas for processing...")
    pdf = df.compute()

    print("Applying undersampling...")
    rus = RandomUnderSampler(random_state=10, sampling_strategy=0.85)
    features = pdf.drop('Label', axis=1)
    
    features = features.reset_index(drop=True)
    X_res, y_res = rus.fit_resample(features, pdf['Label'])
    
    df_resampled = pd.DataFrame(X_res, columns=features.columns)
    df_resampled['Label'] = y_res
    
    print("\nStage 4 data head:")
    print(df_resampled.head())
    print("Saving stage 4 output...")
    dd.from_pandas(df_resampled, npartitions=10).to_parquet(stage4_path)
    print("Stage 4: Complete\n")

def stage5_scaling(paths):
    """Scale features using QuantileTransformer"""
    stage4_path = os.path.join(paths['tmp'], 'stage4.parquet')
    stage5_path = os.path.join(paths['tmp'], 'stage5.parquet')
    transformer_path = os.path.join(paths['transformer'], 'quantile_transformer.pkl')
    
    if os.path.exists(stage5_path):
        print("Stage 5 already completed, skipping...")
        return
    
    print("Stage 5: Loading data from stage 4...")
    df = dd.read_parquet(stage4_path)
    
    print("Scaling features...")
    pdf = df.compute()
    
    feature_cols = [col for col in pdf.columns if col != 'Label']
  
    qt = QuantileTransformer(output_distribution='normal', random_state=10)
    pdf[feature_cols] = qt.fit_transform(pdf[feature_cols])
    
   
    os.makedirs(os.path.dirname(transformer_path), exist_ok=True)
    with open(transformer_path, 'wb') as f:
        pickle.dump(qt, f)
    
    print("\nStage 5 data head:")
    print(pdf.head())
    
    dd.from_pandas(pdf, npartitions=10).to_parquet(stage5_path)
    print("Stage 5: Complete\n")

def stage6_feature_importance(paths):
    """Compute feature importance and selection - more aggressive"""
    stage5_path = os.path.join(paths['tmp'], 'stage5.parquet')
    stage6_path = os.path.join(paths['tmp'], 'stage6.parquet')
    
    if os.path.exists(stage6_path):
        print("Stage 6 output exists, skipping...")
        return dd.read_parquet(stage6_path).compute()
    
    print("Stage 6: Loading data from stage 5...")
    df = dd.read_parquet(stage5_path)
    
    print("Computing feature importance...")
    def compute_importances(df):
        rfc = RandomForestClassifier(
            random_state=10, 
            n_jobs=-1,
            n_estimators=100, 
            max_depth=15
        )
        rfc.fit(df.drop('Label', axis=1), df['Label'])
        return pd.DataFrame({
            'features': df.columns[:-1],
            'importance': rfc.feature_importances_
        })
    
    meta = pd.DataFrame({'features': pd.Series(dtype='str'), 'importance': pd.Series(dtype='float')})
    importances = df.map_partitions(compute_importances, meta=meta).compute()
    importances = importances.groupby('features').mean().sort_values('importance', ascending=False)

    important_features = importances.head(15).index.tolist()
    df = df[important_features + ['Label']]
    
    print("Selected features:", important_features)
    
    print("Saving stage 6 output...")
    df.to_parquet(stage6_path)
    print("\nStage 6 data head:")
    print(df.head())
    print("Stage 6: Complete\n")
    
    return df.compute()

def main():
    dask.config.set(scheduler='threads')
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
    
    paths = setup_paths()
    
    try:

        stage1_initial_processing(paths)
        stage2_preprocessing(paths)
        stage3_encoding(paths)
        stage4_undersampling(paths)
        stage5_scaling(paths)
        df_final = stage6_feature_importance(paths)
        
        print("Final processing...")
        X_train, X_test, y_train, y_test = train_test_split(
            df_final.drop('Label', axis=1),
            df_final['Label'],
            train_size=0.7,
            random_state=10
        )

        X_train.to_csv(os.path.join(paths['output'], 'train_data.csv'), index=False)
        X_test.to_csv(os.path.join(paths['output'], 'test_data.csv'), index=False)
        y_train.to_csv(os.path.join(paths['output'], 'train_labels.csv'), index=False)
        y_test.to_csv(os.path.join(paths['output'], 'test_labels.csv'), index=False)
        
        print("Processing complete!")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        raise

if __name__ == '__main__':
    main()