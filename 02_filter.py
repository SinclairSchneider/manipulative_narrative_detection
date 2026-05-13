import pandas as pd
import json
from tqdm import tqdm

df = pd.read_json("tweets_about_german_politicians_jan_feb_2025_reddit_and_telegram_BY_classified.json")

def get_classification(json_string):
    try:
        json_dict = json.loads(json_string)
        contains_narrative = json_dict.get('contains_narrative', False)
        contains_narrative = bool(contains_narrative)
        reasoning = json_dict.get('reasoning', '')
    except:
        contains_narrative = False
        reasoning = ""
    return [contains_narrative, reasoning]

result = [get_classification(x) for x in tqdm(df['classified'])]

df['contains_narrative'] = [x[0] for x in result]
df['reasoning'] = [x[1] for x in result]

df_filtered = df.drop(columns=['classified'])

df_filtered = df_filtered[df_filtered['contains_narrative']].reset_index(drop=True)

df_filtered.to_json("tweets_about_german_politicians_jan_feb_2025_reddit_and_telegram_BY_classified_filtered.json")