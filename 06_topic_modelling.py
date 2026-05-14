from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.backend import BaseEmbedder
from bertopic.dimensionality import BaseDimensionalityReduction
from bertopic.cluster import BaseCluster
from bertopic.representation import BaseRepresentation
from datasets import load_dataset
from bertopic.representation import OpenAI
import openai
import datamapplot 
from matplotlib.figure import Figure 
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import concurrent.futures

df = pd.read_parquet("tweets_about_german_politicians_jan_feb_2025_reddit_and_telegram_classified_embeddings_umap_cluster.parquet")

class ThreadedReasoningVLLMRepresentation(BaseRepresentation):
    def __init__(self, client, model_id, prompt_template, max_concurrent=10):
        self.client = client
        self.model_id = model_id
        self.prompt_template = prompt_template
        self.max_concurrent = max_concurrent # Number of simultaneous requests to send to vLLM

    def _process_topic(self, topic, topic_words, topic_model, documents):
        """Worker function that processes a single topic."""
        if topic == -1: 
            return topic, None
        
        keywords = [word for word, _ in topic_words]
        
        # Safely get representative documents
        repr_docs = topic_model.get_representative_docs(topic)
        if not repr_docs:
            topic_docs = documents[documents["Topic"] == topic]["Document"].values
            repr_docs = topic_docs[:5] if len(topic_docs) > 0 else [""]
            
        docs_str = "\n".join([f"- {doc}" for doc in repr_docs[:5]])
        
        # Truncate context
        if len(docs_str) > 500000:
            docs_str = docs_str[:500000] + "\n... [TRUNCATED TO FIT CONTEXT]"
            
        keywords_str = ", ".join(keywords)
                
        formatted_prompt = self.prompt_template.replace("[DOCUMENTS]", docs_str).replace("[KEYWORDS]", keywords_str)
        final_label = keywords[0] 
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": "You are a helpful and precise threat intelligence analyst."},
                    {"role": "user", "content": formatted_prompt}
                ],
                max_tokens=200000, # Gives Qwen plenty of room to write its reasoning steps
                temperature=0.1,
                frequency_penalty=0.1
            )
            
            message = response.choices[0].message
            msg_dump = message.model_dump()
            reasoning_text = msg_dump.get("reasoning_content", "")
            
            if message.content is None:
                raw_output = reasoning_text if reasoning_text else ""
            else:
                raw_output = message.content.strip()
                if reasoning_text:
                    raw_output = reasoning_text + "\n" + raw_output
            
            if raw_output:
                for line in raw_output.split('\n'):
                    if line.strip().upper().startswith("LABEL:"):
                        final_label = line.split("LABEL:", 1)[1].strip()
                        break
                        
        except Exception as e:
            print(f"\n[ERROR] Failed to generate label for topic {topic}: {e}")
            print(message)
        
        return topic, [final_label] + keywords

    def extract_topics(self, topic_model, documents, c_tf_idf, topics):
        updated_topics = {}
        
        # Initialize the Thread Pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            # Submit all topics to the executor
            futures = {
                executor.submit(self._process_topic, topic, topic_words, topic_model, documents): topic 
                for topic, topic_words in topics.items()
            }
            
            # Wrap in tqdm to monitor progress as threads complete
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Generating Narrative Labels (Parallel)"):
                topic, result = future.result()
                if result is not None:
                    updated_topics[topic] = result
                    
        return updated_topics

# Step 4 - Tokenize topics
vectorizer_model = CountVectorizer(stop_words = stopwords.words('german'))

# Step 5 - Create topic representation
ctfidf_model = ClassTfidfTransformer()

# Step 6 - (Optional) Fine-tune topic representations with
# 1. Update the client to point to your vLLM server
client = openai.OpenAI(
    base_url='http://127.0.0.1:8002/v1', 
    api_key='vllm', # The OpenAI client requires a string here, even if vLLM ignores it
    timeout=None,
)

# System prompt describes information given to all conversations
prompt = """
You are an expert threat intelligence analyst specializing in Foreign Information Manipulation and Interference (FIMI). Your task is to extract and label the underlying "strategic narrative" from a cluster of social media posts.

DEFINITION OF A STRATEGIC NARRATIVE:
In this context, a narrative is NOT just a descriptive topic (like "economy" or "infrastructure"). It is a strategically constructed storyline designed to manipulate. It typically "collapses uncertainty into a malicious plot". A good narrative label identifies:
1. The core claim (what is allegedly happening).
2. The enemy image or scapegoat (who is to blame).
3. The manipulative angle (e.g., zero-sum fallacy, deliberate betrayal, manufactured crisis).

EXAMPLE:
Documents:
- "Funny how they always find billions for Ukraine, but for our hospitals and pensions there’s ‘no budget.’ The government hates its own people!"
- "Look at the crumbling bridges in our city. That's where your tax money goes while the elites fund their proxy war. This is deliberate destruction of our wealth."
Keywords: 'billions, tax, ukraine, infrastructure, government, elites, destroy, pensions, proxy war, wealth'

Reasoning: The documents consistently contrast domestic hardship (hospitals, bridges) with foreign funding (Ukraine, proxy war). The blamed entity is the "government" or "elites". The manipulation tactic uses a zero-sum fallacy to claim domestic wealth is deliberately destroyed for a foreign war.
Label: Zero-Sum Fallacy: The government is deliberately destroying domestic wealth and local infrastructure to fund a foreign proxy war.

YOUR TASK:
I have a topic that contains the following documents:
[DOCUMENTS]

The topic is described by the following keywords: '[KEYWORDS]'.

Step 1: Provide a brief step-by-step reasoning evaluating the core claim, the enemy image, and the manipulative angle.
Step 2: Formulate a short, precise label of the strategic narrative connecting them. State the manipulative storyline in exactly one concise sentence.
Step 3: Output your final label on a new line starting EXACTLY with the word "LABEL: ".
"""

# Create your representation model
threaded_vllm_model = ThreadedReasoningVLLMRepresentation(
    client=client, 
    #model_id='Qwen/Qwen3.5-122B-A10B-FP8',
    model_id='Qwen/Qwen3.5-397B-A17B-FP8',
    prompt_template=prompt,
    max_concurrent=10
)

representation_model = {"LLM_title": threaded_vllm_model}

#df_test = df.head(1000)

for c in tqdm([x for x in df.columns if "min_cluster_size" in x]):
    empty_embedding_model = BaseEmbedder()
    empty_reduction_model = BaseDimensionalityReduction()
    empty_cluster_model = BaseCluster()
    # All steps together
    topic_model = BERTopic(
        #nr_topics=6,
        embedding_model=empty_embedding_model,     # Step 1 - Extract embeddings
        umap_model=empty_reduction_model,          # Step 2 - Reduce dimensionality
        hdbscan_model=empty_cluster_model,         # Step 3 - Cluster reduced embeddings
        vectorizer_model=vectorizer_model,         # Step 4 - Tokenize topics
        ctfidf_model=ctfidf_model,                 # Step 5 - Extract topic words
        representation_model=representation_model, # Step 6 - (Optional) Fine-tune topic representations
        verbose=True
    )
    
    min_cluster_size = int(c.replace("min_cluster_size_", ""))
    size = max(df[c])
    topic_model_fitted = topic_model.fit(df['text'], embeddings=np.stack(df['umap_5d'].values), y=list(df[c]))
    output_name = "tweets_about_german_politicians_jan_feb_2025_reddit_and_telegram_min_cluster_size_"+str(min_cluster_size)+"_number_of_clusters_"+str(size)
    topic_model_fitted.save(output_name, serialization="safetensors", save_ctfidf=True)
    #print(c+" min cluster size: "+str(min_cluster_size)+" number of clusters: "+str(size))