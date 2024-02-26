from transformers import pipeline
from datasets import load_dataset
from evaluate import evaluator
import evaluate

pipe = pipeline("text-classification", model="lvwerra/distilbert-imdb")
data = load_dataset("imdb", split="test").shuffle().select(range(1000))
metric = evaluate.load("accuracy")

task_evaluator = evaluator("text-classification")

results = task_evaluator.compute(model_or_pipeline=pipe, data=data, metric=metric, label_mapping={"NEGATIVE": 0, "POSITIVE": 1},)

print(results)