from glob import glob
from bertopic import BERTopic
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np

folders = glob("tweets_about_german_politicians_jan_feb_2025_reddit_and_telegram_min_cluster_size_*")
#folders = glob("precomputed_tweets_about_german_politicians_jan_feb_2025_reddit_and_telegram_min_cluster_size_*")

df = pd.read_parquet("tweets_about_german_politicians_jan_feb_2025_reddit_and_telegram_classified_embeddings_umap_cluster.parquet")
reduced_embeds_array = np.array([[float(x) for x in list(x)] for x in df['umap_2d']])

for path in tqdm(folders):
    topic_model = BERTopic.load(path)
    number_of_topics = int(path.split("_")[-1])
    Path(str(number_of_topics)+"_Topics").mkdir(parents=True, exist_ok=True)
    
    topics = [str(topic).strip().split(".")[0] for topic in topic_model.get_topic_info()['LLM_title']]
    topics = [x.replace("Topic: ", "") for x in topics]
    topics = ["<sup>"+" ".join([x+"<br>" if (i+1)%9==0 else x for i, x in enumerate(x.split(" "))]).replace("<br> ", "<br>")+"</sup>" for x in topics]
    topic_model.set_topic_labels(topics)
    visualization = topic_model.visualize_barchart(top_n_topics=100, n_words = 10, custom_labels=True, width=500, height=500, title="Narratives<br>")
    visualization.write_image(str(number_of_topics)+"_Topics/"+str(number_of_topics)+"_Topics_politician_tweets_barchart_narratives.pdf",scale=1, width=2000, height=8000)
    
    fig = topic_model.visualize_documents(df['text'], reduced_embeddings=reduced_embeds_array, hide_document_hover=False, hide_annotations=True, custom_labels=True, title="Narratives")
    fig.write_html(str(number_of_topics)+"_Topics/"+str(number_of_topics)+"_Topics_politician_tweets_visualize_documents_narratives.html")
    
    fig = topic_model.visualize_document_datamap(df['text'], reduced_embeddings=reduced_embeds_array, custom_labels=True, title="Narratives")
    fig.savefig(str(number_of_topics)+"_Topics/"+str(number_of_topics)+"_Topics_politician_tweets_datamap_narratives.pdf", bbox_inches="tight")
    
    fig = topic_model.visualize_document_datamap(df['text'], reduced_embeddings=reduced_embeds_array, interactive=True, custom_labels=True)
    fig.save(str(number_of_topics)+"_Topics/"+str(number_of_topics)+"_Topics_politician_tweets_interactive_datamap_narratives.html")