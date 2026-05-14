import pandas as pd
from hdbscan import HDBSCAN
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import os
# Prevent low-level libraries from hogging all cores inside one task
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1"

def run_hdbscan(min_size, data):
    model = HDBSCAN(
        min_cluster_size=min_size, 
        min_samples=100, # Added to prevent over-filtering
        metric='euclidean', 
        cluster_selection_method='eom', 
        prediction_data=True, 
        core_dist_n_jobs=-1 
    )
    labels = model.fit_predict(data)
    return f"min_cluster_size_{min_size}", labels

if __name__ == "__main__":
    # 1. Load Data
    df = pd.read_parquet("tweets_about_german_politicians_jan_feb_2025_reddit_and_telegram_classified_embeddings_umap.parquet")
    
    # 2. Pre-process embeddings once
    print("Preparing embeddings...")
    embeddings = np.stack(df['umap_5d'].values).astype('float32')

    l_min_sizes = [100, 200, 400, 600, 800, 1000]

    # 3. Parallel Execution
    # n_jobs should be set based on RAM. 
    # HDBSCAN on 1M points can use 10GB+ per core. 
    print("Clustering...")
    results = Parallel(n_jobs=8)(
        delayed(run_hdbscan)(size, embeddings) for size in tqdm(l_min_sizes)
    )

    # 4. Map results back to dataframe
    for column_name, labels in results:
        df[column_name] = labels

    # 5. Save
    df.to_parquet(
        "tweets_about_german_politicians_jan_feb_2025_reddit_and_telegram_classified_embeddings_umap_cluster.parquet", 
        engine="pyarrow", 
        row_group_size=10000, 
        compression="snappy"
    )