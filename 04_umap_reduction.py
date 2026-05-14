import pandas as pd
from umap import UMAP
import numpy as np
from tqdm import tqdm

df = pd.read_parquet("tweets_about_german_politicians_jan_feb_2025_reddit_and_telegram_BY_classified_filtered_BY_embeddings.parquet")

df['embeddings'] = [np.array(x) for x in tqdm(df['embeddings'])]

umap_model_5d = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')
umap_model_2d = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine')

reduced_embeddings_5d = umap_model_5d.fit_transform(np.stack(df['embeddings'].values).astype('float32'))
reduced_embeddings_2d = umap_model_2d.fit_transform(np.stack(df['embeddings'].values).astype('float32'))

df['umap_5d'] = [np.array(x) for x in list(reduced_embeddings_5d)]
df['umap_2d'] = [np.array(x) for x in list(reduced_embeddings_2d)]

df.to_parquet("tweets_about_german_politicians_jan_feb_2025_reddit_and_telegram_classified_embeddings_umap.parquet", engine="pyarrow", row_group_size=10000, compression="snappy")