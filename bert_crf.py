import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from transformers import BertModel
from torchcrf import CRF
import os
import glob
import numpy as np
import time
import random
import argparse
from tqdm import tqdm
import warnings

from model.utils import *
from metric import evaluate_pred_file
from config import tag2idx, idx2tag, max_len, max_node, log_fre

warnings.filterwarnings("ignore")
predict_file = "./output/twitter2015/{}/epoch_{}.txt"
device = torch.device("cuda:0")


def seed_torch(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class BertCRFDataset(Dataset):
    def __init__(self, textdir):
        self.X_files = sorted(glob.glob(os.path.join(textdir, "*_s.txt")))
        self.Y_files = sorted(glob.glob(os.path.join(textdir, "*_l.txt")))
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    def __len__(self):
        return len(self.X_files)

    def __getitem__(self, idx):
        with open(self.X_files[idx], "r", encoding="utf-8") as fr:
            s = fr.readline().split("\t")

        with open(self.Y_files[idx], "r", encoding="utf-8") as fr:
            l = fr.readline().split("\t")

        ntokens = ["[CLS]"]
        label_ids = [tag2idx["CLS"]]
        for word, label in zip(s, l):    # iterate every word
            tokens = self.tokenizer._tokenize(word)    # one word may be split into several tokens
            ntokens.extend(tokens)
            for i, _ in enumerate(tokens):
                label_ids.append(tag2idx[label] if i == 0 else tag2idx["X"])
        ntokens = ntokens[:max_len-1]
        ntokens.append("[SEP]")
        label_ids = label_ids[:max_len-1]
        label_ids.append(tag2idx["SEP"])

        input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
        mask = [1] * len(input_ids)
        segment_ids = [0] * max_len

        pad_len = max_len - len(input_ids)
        rest_pad = [0] * pad_len    # pad to max_len
        input_ids.extend(rest_pad)
        mask.extend(rest_pad)
        label_ids.extend(rest_pad)

        # pad ntokens
        ntokens.extend(["pad"] * pad_len)

        return {
            "ntokens": ntokens,
            "input_ids": input_ids,
            "segment_ids": segment_ids,
            "mask": mask,
            "label_ids": label_ids
        }


def collate_fn(batch):
    input_ids = []
    token_type_ids = []
    attention_mask = []
    label_ids = []
    b_ntokens = []

    for _, example in enumerate(batch):
        b_ntokens.append(example["ntokens"])
        input_ids.append(example["input_ids"])
        token_type_ids.append(example["segment_ids"])
        attention_mask.append(example["mask"])
        label_ids.append(example["label_ids"])
            

    return {
        "b_ntokens": b_ntokens,
        "x": {
            "input_ids": torch.tensor(input_ids).to(device),
            "token_type_ids": torch.tensor(token_type_ids).to(device),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.uint8).to(device)
        },
        "y": torch.tensor(label_ids).to(device)
    }


class BertCRFModel(nn.Module):

    def __init__(self, tag2idx=tag2idx):
        super(BertCRFModel, self).__init__()

        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.crf = CRF(len(tag2idx), batch_first=True)
        self.hidden2tag = nn.Linear(768, len(tag2idx))

    def log_likelihood(self, x, y):
        """
        inter_matrix: batch, max_len, 4
        text_mask: batch, max_len
        """
        crf_mask = x["attention_mask"]
        x = self.bert(**x)[0]
        x = self.hidden2tag(x)
        return -self.crf(x, y, mask=crf_mask)

    def forward(self, x):
        """
        inter_matrix: batch, max_len, 4
        text_mask: batch, max_len
        """
        crf_mask = x["attention_mask"]
        x = self.bert(**x)[0]
        x = self.hidden2tag(x)
        return self.crf.decode(x, mask=crf_mask)


def save_model(model, model_path="./model.pt"):
    torch.save(model.state_dict(), model_path)
    print("Current Best bert-crf model has beed saved!")


def predict(epoch, model, dataloader, mode="val", res=None):
    model.eval()
    with torch.no_grad():
        filepath = predict_file.format(mode, epoch)
        with open(filepath, "w", encoding="utf8") as fw:
            for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Predicting"):
                b_ntokens = batch["b_ntokens"]

                x = batch["x"]
                y = batch["y"]
                output = model(x)

                # write into file
                for idx, pre_seq in enumerate(output):
                    ground_seq = y[idx]
                    for pos, (pre_idx, ground_idx) in enumerate(zip(pre_seq, ground_seq)):
                        if ground_idx == tag2idx["PAD"] or ground_idx == tag2idx["X"] or ground_idx == tag2idx["CLS"] or ground_idx == tag2idx["SEP"]:
                            continue
                        else:
                            predict_tag = idx2tag[pre_idx] if idx2tag[pre_idx] not in [
                                "PAD", "X", "CLS", "SEP"] else "O"
                            true_tag = idx2tag[ground_idx.data.item()]
                            line = "{}\t{}\t{}\n".format(b_ntokens[idx][pos], predict_tag, true_tag)
                            fw.write(line)
        print("=============={} -> {} epoch eval done=================".format(mode, epoch))
        cur_f1 = evaluate_pred_file(filepath)
        to_save = False
        if mode == "val":
            if res["best_f1"] < cur_f1:
                res["best_f1"] = cur_f1
                res["epoch"] = epoch
                to_save = True
            print("current best f1: {}, epoch: {}".format(res["best_f1"], res["epoch"]))
        return to_save


def train(args):
    # make initialized weight of each model replica same
    seed_torch(args.seed)

    train_textdir = os.path.join(args.txtdir, "train")
    train_dataset = BertCRFDataset(textdir=train_textdir)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, collate_fn=collate_fn)

    val_textdir = os.path.join(args.txtdir, "valid")
    val_dataset = BertCRFDataset(textdir=val_textdir)
    val_dataloader = DataLoader(val_dataset, batch_size=args.train_batch_size, collate_fn=collate_fn)

    test_textdir = os.path.join(args.txtdir, "test")
    test_dataset = BertCRFDataset(textdir=test_textdir)
    test_dataloader = DataLoader(test_dataset, batch_size=args.train_batch_size, collate_fn=collate_fn)

    model = BertCRFModel()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.8)

    res = {}
    res["best_f1"] = 0.0
    res["epoch"] = -1
    start = time.time()
    for epoch in range(args.num_train_epoch):
        model.train()
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            x = batch["x"]
            y = batch["y"]

            loss = model.log_likelihood(x, y)
            loss.backward()
            optimizer.step()

            if i % log_fre == 0:
                print("EPOCH: {} Step: {} Loss: {}".format(epoch, i, loss.data))

        scheduler.step()
        to_save = predict(epoch, model, val_dataloader, mode="val", res=res)
        predict(epoch, model, test_dataloader, mode="test", res=res)
        if to_save:    # whether to save the best checkpoint
            save_model(model, args.ckpt_path)

    print("================== train done! ================")
    end = time.time()
    hour = int((end-start)//3600)
    minute = int((end-start)%3600//60)
    print("total time: {} hour - {} minute".format(hour, minute))


def test(args):
    model = BertCRFModel().to(device)
    model.load_state_dict(torch.load(args.ckpt_path))

    test_textdir = os.path.join(args.txtdir, "test")
    test_dataset = BertCRFDataset(textdir=test_textdir)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, collate_fn=collate_fn)

    model.eval()
    with torch.no_grad():
        filepath = "./test_output.txt"
        with open(filepath, "w", encoding="utf8") as fw:
            for i, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="Testing"):
                b_ntokens = batch["b_ntokens"]
                x = batch["x"]
                y = batch["y"]
                output = model(x)

                # write into file
                for idx, pre_seq in enumerate(output):
                    ground_seq = y[idx]
                    for pos, (pre_idx, ground_idx) in enumerate(zip(pre_seq, ground_seq)):
                        if ground_idx == tag2idx["PAD"] or ground_idx == tag2idx["X"] or ground_idx == tag2idx["CLS"] or ground_idx == tag2idx["SEP"]:
                            continue
                        else:
                            predict_tag = idx2tag[pre_idx] if idx2tag[pre_idx] not in [
                                "PAD", "X", "CLS", "SEP"] else "O"
                            true_tag = idx2tag[ground_idx.data.item()]
                            line = "{}\t{}\t{}\n".format(b_ntokens[idx][pos], predict_tag, true_tag)
                            fw.write(line)
        evaluate_pred_file(filepath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument('--txtdir', 
                        type=str, 
                        default="./data/twitter2015/", 
                        help="text dir")
    parser.add_argument('--ckpt_path', 
                        type=str, 
                        default="./model.pt", 
                        help="path to save or load model")
    parser.add_argument("--num_train_epoch",
                        default=30,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--test_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--lr",
                        default=0.0001,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--seed',
                        type=int,
                        default=2021,
                        help="random seed for initialization")
    args = parser.parse_args()
    if args.do_train:
        train(args)
    elif args.do_test:
        test(args)
    else:
        raise ValueError('At least one of `do_train`, `do_eval` must be True.')


if __name__ == "__main__":
    main()
