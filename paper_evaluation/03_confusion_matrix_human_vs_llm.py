import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#Load data
df = pd.read_json("tweets_about_german_politicians_jan_feb_2025_reddit_and_telegram_BY_classified_user.json")
#df = pd.read_json("precomputed_tweets_about_german_politicians_jan_feb_2025_reddit_and_telegram_BY_classified_user.json")
df_filtered = df[df['contains_narrative_user']!="B"].reset_index(drop=True)
df_filtered['contains_narrative_user'] = [True if x=='Y' else False for x in df_filtered['contains_narrative_user']]
df_filtered = pd.concat([df_filtered[df_filtered['contains_narrative']==True][:100],
                         df_filtered[df_filtered['contains_narrative']==False][:100]]).reset_index(drop=True)

#Calculate Matrix
cm = confusion_matrix(df_filtered['contains_narrative_user'], df_filtered['contains_narrative'], labels=[True, False])

#Plotting
sns.set_theme(style="white", font_scale=1.1)
plt.rcParams['font.family'] = 'serif' 
plt.figure(figsize=(8.5, 6.5)) 

cm_percentages = cm / cm.sum(axis=1)[:, np.newaxis]

quadrant_labels = np.array([
    ['Consensus:\nManipulative\nNarrative', 'Divergence:\nHuman Stricter\n'],
    ['Divergence:\nModel Stricter\n', 'Consensus:\nNon-Manipulative\nContent']
])

annot_data = [
    f"{label}\n\n{count}\n({pct:.1%})" 
    for label, count, pct in zip(quadrant_labels.flatten(), cm.flatten(), cm_percentages.flatten())
]
annot_data = np.asarray(annot_data).reshape(2, 2)

ax = sns.heatmap(cm, 
                 annot=annot_data, 
                 fmt='', 
                 cmap='Blues',         
                 cbar=False,           
                 xticklabels=['Manipulative\nNarrative', 'Non-Manipulative\nContent'],
                 yticklabels=['Manipulative\nNarrative', 'Non-Manipulative\nContent'],
                 annot_kws={"size": 11, "weight": "normal"}) 

#Update titles and labels ---
plt.title('Perspective Alignment: Human Evaluation vs. LLM', fontsize=15, pad=20, weight='bold')
plt.xlabel('LLM Classification', fontsize=13, labelpad=12, weight='bold')
plt.ylabel('Human Evaluation', fontsize=13, labelpad=12, weight='bold')

plt.xticks(fontsize=11)
plt.yticks(fontsize=11, rotation=0) 

#Disable border
sns.despine(left=True, bottom=True, top=True, right=True)

plt.tight_layout()
plt.savefig("agreement_matrix_clean.pdf", bbox_inches='tight')
plt.show()