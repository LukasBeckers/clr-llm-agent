from tools.TextNormalizer import TextNormalizer
from algorithms.DynamicTopicModeling import DynamicTopicModeling
import pickle as pk
import os
import time


with open(os.path.join("temp", "dataset"), "rb") as f:
    dataset = pk.load(f)

text_normalizer = TextNormalizer()

for data_point in dataset[:]:
        try:
            data_point["AbstractNormalized"] = text_normalizer(
                data_point["Abstract"]
            )
        except KeyError:
            dataset.remove(data_point)

algo = DynamicTopicModeling(k=6)
start_time = time.time()
results = algo(dataset)
end_time = time.time() - start_time

print("Algorithm Run took: ", end_time, "s ")
print("Results", results)
print(len(results["Topic Words"]))

for x in results["Topic Words"].values():
    print(len(x))
    print(x)