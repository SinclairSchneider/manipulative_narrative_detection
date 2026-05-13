# Readme

## 1.) Clone the repository
```
git clone https://github.com/SinclairSchneider/manipulative_narrative_detection.git
```
## 2.) Get the dataset
```
python3 00_get_data.py
```
## 3.) Classify the dataset

You can adjust CUDA_VISIBLE_DEVICES based on your needs. 

To get more options, run ``python3 01_apply_prompt.py --help``
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 01_apply_prompt.py --model qwen3.5-122b-a10b-fp8 --gpus 4 --dataset tweets_about_german_politicians_jan_feb_2025_reddit_and_telegram.json --output_column_name classified --text_column text
```
The classification process can take from days to weeks, depending on your machine. 

If you don't want to wait, run the following command to download the already classified data:
```
python3 x1_get_classified_data.py
```
## 4.) Apply the classification
This will add a new but smaller file called 

``tweets_about_german_politicians_jan_feb_2025_reddit_and_telegram_BY_classified_filtered.json``

It only contains findings of manipulative posts.
```
python3 02_filter.py
```
