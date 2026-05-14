import numpy as np
import pandas as pd
from glob import glob
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import pdist
from tqdm import tqdm

# 1. Load the same embedding model
print("Loading embedding model...")
model = SentenceTransformer('Qwen/Qwen3-Embedding-8B', trust_remote_code=True)
instruction = 'Identify the strategic narrative, manipulative intent, and underlying disinformation motive in the following text: '

# Get configurations
configurations = glob("../tweets_about_german_politicians_jan_feb_2025_reddit_and_telegram_min_cluster_size_*")
#configurations = glob("../precomputed_tweets_about_german_politicians_jan_feb_2025_reddit_and_telegram_min_cluster_size_*")
configurations = sorted(configurations)[1:] + [sorted(configurations)[0]]

table_data = []

print("Processing configurations...")
for config in tqdm(configurations):
    min_cluster_size = int(config.split("min_cluster_size_")[1].split("_number_of_clusters_")[0])
    
    # Load model and topic info
    topic_model = BERTopic.load(config)
    topic_info = topic_model.get_topic_info()
    
    # 2. Calculate Noise Ratio
    total_posts = topic_info['Count'].sum()
    noise_posts = topic_info[topic_info['Topic'] == -1]['Count'].values[0] if -1 in topic_info['Topic'].values else 0
    noise_ratio = (noise_posts / total_posts) * 100
    
    # 3. Get the valid LLM narrative labels (excluding noise at index 0)
    topics = list(topic_info['LLM_title'])[1:]
    topics = [x.split(".")[0] for x in topics]
    num_clusters = len(topics)
    
    if num_clusters > 1:
        # Prepare the instruction
        texts_to_embed = [instruction + str(topic) for topic in topics]
        
        # Generate embeddings and normalize them
        embeddings = model.encode(texts_to_embed, normalize_embeddings=True)
        
        # Calculate pairwise cosine distances
        # pdist with 'cosine' returns 1 - cosine_similarity
        distances = pdist(embeddings, metric='cosine')
        avg_semantic_distance = np.mean(distances)
    else:
        avg_semantic_distance = 0.0

    table_data.append({
        "min_cluster_size": min_cluster_size,
        "Num Clusters": num_clusters,
        "Noise Ratio (%)": round(noise_ratio, 2),
        "Avg Semantic Distance": round(avg_semantic_distance, 4)
    })

# Print the final dataframe
df_table = pd.DataFrame(table_data)
# Sort by min_cluster_size for clean reading
df_table = df_table.sort_values(by="min_cluster_size").reset_index(drop=True)

print("\n--- RESULTS ---")
print(df_table.to_markdown(index=False))