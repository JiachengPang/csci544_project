import csv
import datasets
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from transformers import BertModel, BertTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

load_embeddings = True
# load_models = True



# dataset = datasets.load_dataset("/openbayes/home/snli")
dataset = datasets.load_dataset("stanfordnlp/snli")


train = dataset["train"]
validation = dataset["validation"]
test = dataset["test"]


model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

model.to(device)

test_entailment = test.filter(lambda x: x["label"] == 0)
test_neutral = test.filter(lambda x: x["label"] == 1)
test_contradiction = test.filter(lambda x: x["label"] == 2)

validation = validation.filter(lambda x: x["label"] != -1)

train = train.filter(lambda x: x["label"] != -1)

test = test.filter(lambda x: x["label"] != -1)

batch_size = 256

test_entailment_ids = np.where(np.array(test["label"]) == 0)[0]
test_neutral_ids = np.where(np.array(test["label"]) == 1)[0]
test_contradiction_ids = np.where(np.array(test["label"]) == 2)[0]


def extract_emb(dataset, _batch_size=batch_size):
    # texts = [data["premise"] + " " + data["hypothesis"] for data in dataset]
    total_samples = len(dataset)

    result = torch.zeros(total_samples, 768)

    for i in range(0, total_samples, _batch_size):
        batch_premise = [data["premise"] for data in dataset[i : i + _batch_size]]
        batch_hypothesis = [data["hypothesis"] for data in dataset[i : i + _batch_size]]

        inputs = tokenizer(
            text=batch_premise,
            text_pair=batch_hypothesis,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu()

        result[i : i + len(batch_premise), :] = cls_embeddings

    return result


# Load embeddings
if load_embeddings:
    test_entailment_emb = torch.from_numpy(np.load("test_entailment_emb.npy"))
    test_neutral_emb = torch.from_numpy(np.load("test_neutral_emb.npy"))
    test_contradiction_emb = torch.from_numpy(np.load("test_contradiction_emb.npy"))
    validation_emb = torch.from_numpy(np.load("validation_emb.npy"))
    train_emb = torch.from_numpy(np.load("train_emb.npy"))
else:
    test_entailment_emb = extract_emb(test_entailment)
    test_neutral_emb = extract_emb(test_neutral)
    test_contradiction_emb = extract_emb(test_contradiction)

    validation_emb = extract_emb(validation)
    train_emb = extract_emb(train)

    # Save embeddings
    np.save("test_entailment_emb.npy", test_entailment_emb.numpy())
    np.save("test_neutral_emb.npy", test_neutral_emb.numpy())
    np.save("test_contradiction_emb.npy", test_contradiction_emb.numpy())
    np.save("validation_emb.npy", validation_emb.numpy())
    np.save("train_emb.npy", train_emb.numpy())

pca = PCA(n_components=64)

test_entailment_emb_pca = pca.fit_transform(test_entailment_emb)
# print(sum(pca.explained_variance_ratio_))
test_neutral_emb_pca = pca.fit_transform(test_neutral_emb)
# print(sum(pca.explained_variance_ratio_))
test_contradiction_emb_pca = pca.fit_transform(test_contradiction_emb)
# print(sum(pca.explained_variance_ratio_))

validation_emb_pca = pca.fit_transform(validation_emb)
# print(sum(pca.explained_variance_ratio_))
train_emb_pca = pca.fit_transform(train_emb)


# k_clusters = int(
#     sum((len(test_entailment), len(test_neutral), len(test_contradiction)))
#     * 0.02
#     // (2 * 3 - 1)
# )
k_clusters = 40

kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init="auto")

test_entailment_labels = kmeans.fit_predict(test_entailment_emb)
test_neutral_labels = kmeans.fit_predict(test_neutral_emb)
test_contradiction_labels = kmeans.fit_predict(test_contradiction_emb)


n_models = 4

# models = {}

# class Classifier(torch.nn.Module):
#     def __init__(self):
#         super(Classifier, self).__init__()
#         self.gru = torch.nn.GRU(64, 64, 10, bidirectional=True)
#         self.dropout = torch.nn.Dropout(0.1)
#         self.fc1 = torch.nn.Linear(128, 64)
#         self.activation = torch.nn.ReLU()
#         self.fc2 = torch.nn.Linear(64, 3)

#     def forward(self, x):
#         x, _ = self.gru(x)
#         x = self.dropout(x)
#         x = self.fc1(x)
#         x = self.activation(x)
#         x = self.fc2(x)
#         return x
# if load_models:
#     for i in range(n_models):
#         models[i] = torch.load(f"model_{i}.pt")
# else:
#     for i in range(n_models):
#         model_ = Classifier()
#         model_.to(device)
#         optimizer = torch.optim.AdamW(model_.parameters(), lr=1e-5)
#         criterion = torch.nn.CrossEntropyLoss()

#         best_accuracy = 0
#         best_loss = 10000

#         early_stop_cnt = 0
#         prev_acc = 0

#         for epoch in range(100):
#             train_loss = 0
#             eval_loss = 0
#             model_.train()
#             for j in range(0, len(train_emb_pca), batch_size):
#                 batch = train_emb_pca[j : j + batch_size]
#                 batch = torch.tensor(batch, dtype=torch.float32).to(device)
#                 labels = torch.tensor(train[j : j + batch_size]["label"]).to(device)

#                 optimizer.zero_grad()
#                 outputs = model_(batch)
#                 loss = criterion(outputs, labels)
#                 loss.backward()
#                 optimizer.step()
#                 train_loss += loss.item()

#             print(
#                 f"Model: {i + 1} *** Epoch {epoch} *** Loss: {train_loss / (len(train_emb_pca) / batch_size)}"
#             )

#             correct = 0
#             total = 0
#             model.eval()

#             with torch.no_grad():
#                 for j in range(0, len(validation_emb_pca), batch_size):
#                     batch = validation_emb_pca[j : j + batch_size]
#                     batch = torch.tensor(batch, dtype=torch.float32).to(device)
#                     labels = torch.tensor(validation[j : j + batch_size]["label"]).to(device)

#                     output= model_(batch)
#                     loss = criterion(output, labels)
#                     eval_loss += loss.item()
#                     _, predicted = torch.max(output.data, 1)
#                     total += labels.size(0)
#                     # print(predicted)
#                     # print(output.shape)
#                     # print(np.argmax(predicted.cpu().detach().numpy(), axis=1).shape)
#                     correct += (predicted.cpu().detach().numpy() == labels.cpu().detach().numpy()).sum().item()

#             # if eval_loss / (len(validation_emb_pca) / batch_size) < best_loss:
#             #     best_loss = eval_loss / (len(validation_emb_pca) / batch_size)
#             #     models[i] = model_.state_dict()
#             acc = 100 * correct / total
#             if acc <= prev_acc:
#                 early_stop_cnt += 1
#             else:
#                 prev_acc = acc
#                 early_stop_cnt = 0

#             if acc > best_accuracy:
#                 best_accuracy = acc
#                 models[i] = model_.state_dict()

#             print(
#                 f"Model: {i + 1} *** Epoch {epoch} *** Eval Loss: {eval_loss / (len(validation_emb_pca) / batch_size)}"
#             )
#             print(f"Accuracy: {100 * correct / total:.4f}")
#             if early_stop_cnt >= 5:
#                 break
#         print(f"Best acc: {best_accuracy}")


#     for i in range(n_models):
#         torch.save(models[i], f"model_{i}.pt")


confidence = np.zeros((len(test), n_models))


for i in range(n_models):
    model_path = f"/openbayes/input/input{i}/"
    model_ = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer_ = AutoTokenizer.from_pretrained(model_path)
    model_.to(device)
    model_.eval()
    with torch.no_grad():
        for j in range(0, len(test), batch_size):
            batch_premise = test["premise"][j : j + batch_size]
            batch_hypothesis = test["hypothesis"][j : j + batch_size]

            inputs = tokenizer_(
                text=batch_premise,
                text_pair=batch_hypothesis,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,  # Use a reasonable max_length
            )

            inputs = {key: value.to(device) for key, value in inputs.items()}

            labels = torch.tensor(test["label"][j : j + batch_size]).to(device)

            model_output = model_(**inputs)

            output = torch.nn.functional.softmax(model_output.logits, dim=1)

            confidence[j : j + labels.size(0), i] = output[range(labels.size(0)), labels].cpu().numpy()


test_entailment_cnt = np.zeros(k_clusters)
test_neutral_cnt = np.zeros(k_clusters)
test_contradiction_cnt = np.zeros(k_clusters)

for i in range(k_clusters):
    test_entailment_cnt[i] = np.sum(test_entailment_labels == i)
    test_neutral_cnt[i] = np.sum(test_neutral_labels == i)
    test_contradiction_cnt[i] = np.sum(test_contradiction_labels == i)
rankings = {
    "entailment": {},
    "neutral": {},
    "contradiction": {},
}

for i in range(k_clusters):
    rankings["entailment"][i] = {}
    rankings["neutral"][i] = {}
    rankings["contradiction"][i] = {}

    for j in range(n_models):
        temp_confidence = confidence[
            test_entailment_ids[test_entailment_labels == i], j
        ]
        temp_rank_id = np.argsort(-temp_confidence)
        temp_rank = temp_rank_id.argsort()
        rankings["entailment"][i][j] = temp_rank

        temp_confidence = confidence[test_neutral_ids[test_neutral_labels == i], j]
        temp_rank_id = np.argsort(-temp_confidence)
        temp_rank = temp_rank_id.argsort()
        rankings["neutral"][i][j] = temp_rank

        temp_confidence = confidence[
            test_contradiction_ids[test_contradiction_labels == i], j
        ]
        temp_rank_id = np.argsort(-temp_confidence)
        temp_rank = temp_rank_id.argsort()
        rankings["contradiction"][i][j] = temp_rank

scores = {
    "entailment": {},
    "neutral": {},
    "contradiction": {},
}

for i in range(k_clusters):
    scores["entailment"][i] = test_entailment_cnt[i] * n_models
    scores["neutral"][i] = test_neutral_cnt[i] * n_models
    scores["contradiction"][i] = test_contradiction_cnt[i] * n_models
    for j in range(n_models):
        scores["entailment"][i] -= rankings["entailment"][i][j]
        scores["neutral"][i] -= rankings["neutral"][i][j]
        scores["contradiction"][i] -= rankings["contradiction"][i][j]


with open("scores.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["entailment", "neutral", "contradiction"])
    for i in range(k_clusters):
        writer.writerow(
            [scores["entailment"][i], scores["neutral"][i], scores["contradiction"][i]]
        )


with open("cluster_indices.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["entailment", "neutral", "contradiction"])
    for i in range(k_clusters):
        writer.writerow(
            [
                test_entailment_ids[test_entailment_labels == i],
                test_neutral_ids[test_neutral_labels == i],
                test_contradiction_ids[test_contradiction_labels == i],
            ]
        )


#  Save confidence in csv

with open("confidence.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow([f"model_{i}" for i in range(n_models)])
    for i in range(len(test)):
        writer.writerow(confidence[i])

# Save rankings in csv

with open("rankings.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["entailment", "neutral", "contradiction"])
    for i in range(k_clusters):
        r_e = 0
        r_n = 0
        r_c = 0

        for j in range(n_models):
            r_e += rankings["entailment"][i][j]
            r_n += rankings["neutral"][i][j]
            r_c += rankings["contradiction"][i][j]

        writer.writerow(
            [
                r_e,
                r_n,
                r_c,
            ]
        )
