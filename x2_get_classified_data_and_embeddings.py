from datasets import load_dataset

df = load_dataset("SinclairSchneider/tweets_about_german_politicians_jan_feb_2025_reddit_telegram_classified_embed_reduced_clustered", split="train").to_pandas()
df = df.drop(columns=['umap_5d', 'umap_2d', 'min_cluster_size_100', 'min_cluster_size_200', 'min_cluster_size_400', 'min_cluster_size_600', 'min_cluster_size_800', 'min_cluster_size_1000'])
df.to_parquet("tweets_about_german_politicians_jan_feb_2025_reddit_and_telegram_BY_classified_filtered_BY_embeddings.parquet")