import re

import numpy as np
import pandas as pd

# Read cluster indices and scores using pandas

scores_df = pd.read_csv("scores.csv")
# print(scores_df.shape)

cluster_indices_df = pd.read_csv("cluster_indices.csv")
# print(cluster_indices_df.shape)

rankings_df = pd.read_csv("rankings.csv")
# print(rankings_df.shape)

low_score_ids = []
k_clusters = 40


def convert_to_list(x):
    #  remove all non-numeric characters, split by space and convert to int
    x = re.sub(r"[^0-9 ]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    x = x.split(" ")
    return np.array(x, dtype=int)


# indices = np.argpartition(-cos_sim, top_k)[:top_k]
for i in range(k_clusters):
    for j in ["entailment", "neutral", "contradiction"]:
        scores_i = convert_to_list(scores_df[j][i])
        ids_i = convert_to_list(cluster_indices_df[j][i])
        rankings_i = convert_to_list(rankings_df[j][i])
        # print(len(scores_i), len(ids_i), len(rankings_i))
        assert len(scores_i) == len(ids_i)
        low_k = min(len(rankings_i) - 1, 4)
        lowest_k = np.argpartition(-rankings_i, low_k)[:low_k]
        low_score_ids.extend(ids_i[lowest_k])

intersection = pd.read_csv("intersection_output_ignore_prediction.csv")

intersection_ids = intersection["index"].to_list()

#  intersection of low_score_ids and intersection_ids
low_score_intersection = list(set(low_score_ids).intersection(intersection_ids))
