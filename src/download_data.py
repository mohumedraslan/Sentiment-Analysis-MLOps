from datasets import load_dataset
dataset = load_dataset("imdb", split="train")
dataset.to_csv("data/imdb_small.csv")