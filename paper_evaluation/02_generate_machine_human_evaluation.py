import pandas as pd
import os
import numpy as np
from tqdm import tqdm

OUTPUT_PATH = "tweets_about_german_politicians_jan_feb_2025_reddit_and_telegram_BY_classified_user.json"
RANDOM_STATE = 42

def show_stats(existing_df):
    total = len(existing_df)
    print(f"\n{'='*60}")
    print(f"STATISTICS ({total} rows saved):")
    if 'contains_narrative' in existing_df.columns:
        print(f"  contains_narrative=True:  {int((existing_df['contains_narrative']==True).sum())}")
        print(f"  contains_narrative=False: {int((existing_df['contains_narrative']==False).sum())}")
    else:
        print(f"  contains_narrative: N/A (column not in output file)")
    print(f"  contains_narrative_user=True:                {int((existing_df['contains_narrative_user'].isin(['Y'])).sum()) if total > 0 else 0}")
    print(f"  contains_narrative_user=False:               {int((existing_df['contains_narrative_user'].isin(['N'])).sum()) if total > 0 else 0}")
    print(f"  contains_narrative_user=Borderline:          {int((existing_df['contains_narrative_user'].isin(['B'])).sum()) if total > 0 else 0}")
    print(f"  coherence_user=True:                         {int((existing_df['coherence_user']==True).sum()) if total > 0 else 0}")
    print(f"  coherence_user=False:                        {int((existing_df['coherence_user']==False).sum()) if total > 0 else 0}")
    print(f"{'='*60}")

def get_balanced_queue(solved_keys, df):
    # Get all unsolved rows
    unsolved = [(idx, row) for idx, row in df.iterrows() if (row['id'], row['source']) not in solved_keys]

    if not unsolved:
        return []

    # Split by contains_narrative
    true_rows = [(idx, row) for idx, row in unsolved if row.get('contains_narrative') == True]
    false_rows = [(idx, row) for idx, row in unsolved if row.get('contains_narrative') == False]

    print(f"  Unsolved: {len(true_rows)} True, {len(false_rows)} False")

    # Undersample the larger group to match the smaller
    min_len = min(len(true_rows), len(false_rows))
    rng = np.random.RandomState(RANDOM_STATE)

    true_sampled = rng.choice(len(true_rows), size=min_len, replace=False).tolist()
    true_balanced = [true_rows[i] for i in true_sampled]

    false_sampled = rng.choice(len(false_rows), size=min_len, replace=False).tolist()
    false_balanced = [false_rows[i] for i in false_sampled]

    # Shuffle together
    combined = true_balanced + false_balanced
    rng.shuffle(combined)
    return combined

#df = pd.read_json("../tweets_about_german_politicians_jan_feb_2025_reddit_and_telegram_BY_classified_small.json")
df = pd.read_json("../tweets_about_german_politicians_jan_feb_2025_reddit_and_telegram_BY_classified.json")

if 'text' not in df.columns:
    raise ValueError("Text column is missing")

before = len(df)
df = df[df['text'].str.strip().str.len() > 0]
print(f"Filtered out {before - len(df)} empty/whitespace texts: {len(df)} rows left")

# Check which rows have already been solved by reading output file
solved_keys = set()
if os.path.exists(OUTPUT_PATH):
    existing_df = pd.read_json(OUTPUT_PATH)
    for col in ['contains_narrative_user', 'coherence_user']:
        if col not in existing_df.columns:
            existing_df[col] = None
    solved_df = existing_df[
        existing_df['contains_narrative_user'].notna() &
        existing_df['coherence_user'].notna()
    ]
    solved_keys = set(zip(solved_df['id'], solved_df['source']))
    print(f"Loaded {len(solved_keys)} previously solved rows from output file.")
    show_stats(existing_df)

queue = get_balanced_queue(solved_keys, df)

if not queue:
    print("All rows have been solved!")
else:
    print(f"Starting {len(queue)}/{len(df)} rows (balanced True/False, shuffled).")

    for idx, row in tqdm(queue, total=len(queue)):
        # Step 1: Ask about narrative
        print(f"\n{'='*60}")
        print(f"Row {idx}")
        print(f"Text: {row['text']}")
        narrative = input("Is this a narrative? (Y/N/B): ").strip().upper()

        # Step 2: Ask about coherence
        contains_narrative = row.get('contains_narrative', 'N/A')
        reasoning = row.get('reasoning', 'N/A')
        print(f"\ncontains_narrative: {contains_narrative}")
        print(f"reasoning: {reasoning}")
        coherence = input("Is it coherent with contains_narrative? (Y/N): ").strip().upper()
        coherence = True if coherence == 'Y' else False

        # Build solved row
        solved_row = row.to_dict()
        solved_row['contains_narrative_user'] = narrative
        solved_row['coherence_user'] = coherence

        # Save only solved rows (append to existing)
        solved_rows_list = []
        if os.path.exists(OUTPUT_PATH):
            existing_df = pd.read_json(OUTPUT_PATH)
            solved_rows_list = existing_df.to_dict(orient='records')

        solved_rows_list.append(solved_row)
        pd.DataFrame(solved_rows_list).to_json(OUTPUT_PATH, orient='records', indent=2)

        print(f"Saved! {len(solved_rows_list)} of {len(df)} rows saved.")

        # Show updated statistics after each save
        show_stats(pd.DataFrame(solved_rows_list))
