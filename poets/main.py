import copy
import random
import data
import nn_config
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import pandas as pd
import time
import os
import json
from common import text_utils
from poets import AuthorNet as an
from poets import datasets
from types import SimpleNamespace
import winsound

PAD = 0

for SEED in nn_config.SEED:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Read data
    filtered_posts_df = data.load_filtered_posts()
    # Save to CSV for debugging.
    # filtered_posts_df.to_csv("filtered_posts_df.csv", index=False, encoding="utf-8-sig")

    # prepare coding dictionaries
    train_set_df = filtered_posts_df[filtered_posts_df["split"] == "train"].copy()
    char_set = ["PAD", "UNK"] + sorted(set(train_set_df["masked_text"].str.cat(sep=" ")))
    id_to_char = {index: value for index, value in enumerate(char_set)}
    char_to_id = {value: index for index, value in enumerate(char_set)}
    authors = sorted(train_set_df["PageName"].unique())
    author_to_id = {name: i for i, name in enumerate(authors)}
    id_to_author = {i: name for i, name in enumerate(authors)}
    num_authors = len (authors)

    nnconf = {k: v for k, v in vars(nn_config).items() if not k.startswith("_")}
    nnconf['char_to_id'] = char_to_id
    nnconf['author_to_id'] = author_to_id
    nnconf["SEED"] = SEED
    nnconf_obj = SimpleNamespace(**nnconf)

    train_set_df["coded_text"] = text_utils.conv_char_to_code(train_set_df["masked_text"], char_to_id)
    train_set_df["coded_lines"] = text_utils.conv_char_to_code(train_set_df["lines"], char_to_id)

    # testing
    # train_set_df["decoded_text"] = text_utils.conv_code_to_char(train_set_df["coded_text"], id_to_char)
    # print((train_set_df["decoded_text"] == train_set_df["masked_text"]).all())

    train_data_1D = train_set_df["coded_text"].tolist()
    train_data_2D = train_set_df["coded_lines"].tolist()
    train_labels = torch.tensor(train_set_df["PageName"].map(author_to_id).tolist(), dtype=torch.long)
    train_set = datasets.PostDataset(train_data_1D, train_data_2D, train_labels, nnconf_obj)
    train_dataloader = DataLoader(train_set, batch_size=32, shuffle=False)

    # Validation setup
    val_set_df = filtered_posts_df[filtered_posts_df["split"] == "val"].copy()
    val_set_df["coded_text"] = text_utils.conv_char_to_code(val_set_df["masked_text"],char_to_id)
    val_set_df["coded_lines"] = text_utils.conv_char_to_code(val_set_df["lines"],char_to_id)

    val_data_1d = val_set_df["coded_text"].tolist()
    val_data_2d = val_set_df["coded_lines"].tolist()
    val_labels = torch.tensor(val_set_df["PageName"].map(author_to_id).tolist(), dtype=torch.long)
    val_set = datasets.PostDataset(val_data_1d, val_data_2d, val_labels, nnconf_obj, VALIDATION=True)
    val_dataloader = DataLoader(val_set, batch_size=32, shuffle=False)


    def run_validation(model, criterion_val, val_dataloader):
        model.eval()
        m={}
        correct = 0
        total = 0
        logits_list= []
        labels_list = []
        predicted_list = []
        with torch.no_grad():
            val_loss = 0.0
            conf_mat = np.zeros([num_authors,num_authors],dtype=np.int64)
            for inputs_1D, inputs_2D, labels in val_dataloader:
                outputs = model(inputs_1D, inputs_2D)
                loss = criterion_val(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                for p, l in zip(predicted, labels):
                    conf_mat[l, p] += 1
                logits_list.append(outputs)
                labels_list.append(labels)
                predicted_list.append(predicted)

        m['recall_per_author'] = (np.diag(conf_mat) / conf_mat.sum(axis=1))
        m['balanced_accuracy'] = m['recall_per_author'].mean()
        m['accuracy'] = 100 * correct / total
        m['val_loss_avg'] = val_loss/len(val_dataloader)
        m['conf_mat'] = conf_mat
        m['logits'] = torch.cat(logits_list)
        m['labels'] = torch.cat(labels_list)
        m['predicted'] = torch.cat(predicted_list)
        m['probabilities'] = torch.softmax(m['logits'], dim=1)

        return m

    model = an.AuthorNet(SimpleNamespace(**nnconf))
    criterion_train = nn.CrossEntropyLoss()
    criterion_val = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = np.inf
    best_balanced_accuracy = -np.inf
    best_model_state = copy.deepcopy(model.state_dict())
    best_accuracy = 0
    start_time = time.perf_counter()
    last_improve_epoch = -1
    for epoch in range(nn_config.MAX_EPOCHS):
        train_set.set_epoch()
        model.train()
        train_loss = 0.0
        b = 0
        for inputs_1D, inputs_2D, labels in train_dataloader:
            optimizer.zero_grad()           # איפוס הגרדיאנטים
            outputs = model(inputs_1D, inputs_2D)          # Forward Pass
            loss = criterion_train(outputs, labels) # חישוב המחיר
            loss.backward()                  # Backward Pass (חישוב גרדיאנטים)
            optimizer.step()                 # עדכון משקולות
            train_loss += loss.item()

        print(f"Epoch {epoch+1} - Loss: {train_loss/len(train_dataloader)}")

        metrics = run_validation(model, criterion_val, val_dataloader)
        print(f"Validation Accuracy: {metrics['accuracy']:.2f}%  Loss: {metrics['val_loss_avg']}  Balanced Accuracy: {metrics['balanced_accuracy'] * 100:.2f}%")
    #    if val_loss_avg < best_val_loss - nn_config.min_delta:
        if metrics['balanced_accuracy'] > best_balanced_accuracy + nn_config.MIN_DELTA:
            best_val_loss = metrics['val_loss_avg']
            best_balanced_accuracy = metrics['balanced_accuracy']
            last_improve_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            best_accuracy = metrics['accuracy']
        elif epoch - last_improve_epoch >= nn_config.PATIENCE:
            break

    train_time = time.perf_counter() - start_time
    print(20*"=")
    print(f"SEED={SEED}")
    print(f"Training time: {train_time:.2f} seconds")
    print(f"Selected epoch {last_improve_epoch + 1} from {epoch + 1}")
    print("Training Finished!")

    start_time = time.perf_counter()
    model.load_state_dict(best_model_state)
    metrics = run_validation(model, criterion_val, val_dataloader)

    val_time = time.perf_counter() - start_time
    print(f"Validation time: {val_time:.2f} seconds")
    print(f"Validation loss: {metrics['val_loss_avg']} Balanced accuracy: {metrics['balanced_accuracy']} \n Confusion Matrix:")
    conf_df = pd.DataFrame(metrics["conf_mat"], index=authors, columns=range(num_authors) )
    print("     " + " ".join(f"{i:3d}" for i in range(num_authors)))

    for i, row in enumerate(metrics["conf_mat"]):
        numbers = " ".join(f"{n:3d}" for n in row)
        print(f"{chr(65 + i)}   {numbers}   {authors[i]}")
    print(20*"=")
    print("Validation Finished!")


    metrics['post_ids'] = val_set_df["Post_ID"]
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    directory = "experiments/" + timestamp
    os.makedirs(directory, exist_ok=False)
    torch.save(model.state_dict(), os.path.join(directory,"model.pth"))
    with open(os.path.join(directory,'netconfig.json'), 'w') as f:
        json.dump(nnconf, f, indent=4)
    np.savez_compressed(os.path.join(directory,'validation.npz'),
                        recall_per_author=metrics['recall_per_author'],
                        balanced_accuracy=metrics['balanced_accuracy'],
                        accuracy=metrics['accuracy'],
                        val_loss_avg = metrics['val_loss_avg'],
                        logits=metrics['logits'].cpu().numpy(),
                        labels=metrics['labels'].cpu().numpy(),
                        predicted = metrics['predicted'].cpu().numpy(),
                        probabilities=metrics['probabilities'].cpu().numpy(),
                        post_ids=metrics['post_ids'],
                        masked_length=val_set_df["masked_length"].to_numpy(),
                        conf_mat=metrics['conf_mat'],
                        epochs=epoch + 1,
                        selected_epoch=last_improve_epoch + 1,
                        train_time=train_time
                        )

    model.eval()

    # Append experiment summary to CSV
    results_file = os.path.join("experiments", "results.csv")

    config_params = {
        k: v for k, v in vars(nn_config).items()
        if not k.startswith("_")
    }
    config_params["SEED"] = SEED

    result_row = {
        "timestamp": timestamp,
        **config_params,
        "epochs": epoch + 1,
        "selected_epoch": last_improve_epoch + 1,
        "train_time_s": round(train_time, 2),
        "val_loss": metrics["val_loss_avg"],
        "accuracy_pct": metrics["accuracy"],
        "balanced_accuracy_pct": metrics["balanced_accuracy"] * 100,
    }

    new_result = pd.DataFrame([result_row])

    if os.path.exists(results_file):
        previous_results = pd.read_csv(results_file)
        new_result = pd.concat(
            [previous_results, new_result],
            ignore_index=True
        )

    new_result.to_csv(
        results_file,
        index=False,
        encoding="utf-8-sig"
    )

winsound.Beep(400, 500)

# model = AuthorNet()
# model.load_state_dict(torch.load("best_model.pth", weights_only=True))
# model.eval()
