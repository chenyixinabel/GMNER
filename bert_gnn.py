import os, argparse, math, glob, random, pickle, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchcrf import CRF
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter
from tqdm import tqdm
from model.utils import GELU, make_one_hot
from config import tag2idx, idx2tag, log_fre
from metric import evaluate_pred_file
from graph import tweet_preprocess, label_preprocess

device = torch.device("cuda:0")
predict_file = "./output/twitter2015/{}/epoch_{}.txt"

vocab2vec = None

def seed_torch(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class BERTGNN_Dataset(Dataset):
    def __init__(self, text_dir, vocab2vec):
        self.X_files = sorted(glob.glob(os.path.join(text_dir, "*_s.txt")), key=lambda x: int(x.split('/')[-1].split('_')[0]))
        self.Y_files = sorted(glob.glob(os.path.join(text_dir, "*_l.txt")), key=lambda x: int(x.split('/')[-1].split('_')[0]))
        self.vocab2vec = vocab2vec
    
    def __len__(self):
        return len(self.X_files)
    
    def __getitem__(self, idx):
        with open(self.X_files[idx], "r", encoding="utf-8") as fr:
            words = fr.readline().split("\t")
            sent = " ".join(words)

        with open(self.Y_files[idx], "r", encoding="utf-8") as fr:
            labels = fr.readline().split("\t")

        tokens, token_vecs, edge_index, edge_type = tweet_preprocess(sent, self.vocab2vec)
        labels_ext = label_preprocess(words, sent, labels)
        label_ids = [tag2idx[l] for l in labels_ext]

        return {
            "tokens": tokens,
            "token_vecs": token_vecs,
            "edge_index": edge_index,
            "edge_type": edge_type,
            "label_ids": label_ids
        }

def collate_fn(batch):
    b_tokens = []
    token_vecs = []
    edge_index = []
    edge_type = []
    label_ids = []

    sample_length_cum = 0
    for sample in batch:
        b_tokens += sample["tokens"]
        token_vecs.append(sample["token_vecs"])
        edge_index.append(sample["edge_index"] + sample_length_cum)
        edge_type.append(sample["edge_type"])
        label_ids += sample["label_ids"]
        sample_length_cum += sample["token_vecs"].size(0)

    return {
        "b_tokens": b_tokens,
        "token_vecs": torch.cat(token_vecs, dim=0).to(device),
        "edge_index": torch.cat(edge_index, dim=1).to(device),
        "edge_type": torch.cat(edge_type, dim=0).to(device),
        "label_ids": torch.tensor(label_ids).to(device)
    }

class BERTGNN(nn.Module):
    def __init__(self, n_gnn_layer, n_etype, input_dim, gnn_dim, dropout=0.1):
        super(BERTGNN, self).__init__()
        self.n_gnn_layer = n_gnn_layer
        self.lm2gnn = nn.Linear(input_dim, gnn_dim)
        self.gnn_layers = nn.ModuleList([GATConvE(n_etype, gnn_dim) for _ in range(self.n_gnn_layer)])
        self.feature_orig = nn.Linear(gnn_dim, gnn_dim)
        self.feature_comp = nn.Linear(gnn_dim, gnn_dim)
        self.activation = GELU()
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(gnn_dim, len(tag2idx))
        self.crf = CRF(len(tag2idx), batch_first=True)
        
    def forward(self, node_emb, edge_index, edge_type, layer_dropout_rate=0.1):
        """
        node_emb: (token_vocab_size, input_dim)
        edge_index: list of batch_size, each entry is torch.tensor(2, E)
        edge_type: list of batch_size, each entry is torch.tensor(E, )
        returns: (batch_size, max_len)
        """
        H = self.activation(self.lm2gnn(node_emb))
        X = H
        for _ in range(self.n_gnn_layer):
            X = self.gnn_layers[_](X, edge_index, edge_type)
            X = self.activation(X)
            X = F.dropout(X, layer_dropout_rate, training=self.training)

        hidden = self.feature_orig(H) + self.feature_comp(X)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)

        tag = self.hidden2tag(hidden)
        return self.crf.decode(tag.unsqueeze(0))

    def log_likelihood(self, node_emb, edge_index, edge_type, tag_gt, layer_dropout_rate=0.1):
        """
        node_emb: (token_vocab_size, input_dim)
        edge_index: list of batch_size, each entry is torch.tensor(2, E)
        edge_type: list of batch_size, each entry is torch.tensor(E, )
        returns: (batch_size, max_len)
        """
        H = self.activation(self.lm2gnn(node_emb))
        X = H
        for _ in range(self.n_gnn_layer):
            X = self.gnn_layers[_](X, edge_index, edge_type)
            X = self.activation(X)
            X = F.dropout(X, layer_dropout_rate, training=self.training)

        hidden = self.feature_orig(H) + self.feature_comp(X)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)

        tag = self.hidden2tag(hidden)
        return -self.crf(tag.unsqueeze(0), tag_gt.unsqueeze(0))

class GATConvE(MessagePassing):
    def __init__(self, n_etype, gnn_dim, head_count=4, aggr="add"):
        super(GATConvE, self).__init__(aggr=aggr)
        self.n_etype = n_etype
        self.head_count = head_count
        assert gnn_dim % head_count == 0
        self.dim_per_head = gnn_dim // self.head_count

        self.edge_encoder = nn.Sequential(nn.Linear(self.n_etype+1, gnn_dim), nn.BatchNorm1d(gnn_dim), nn.ReLU(), nn.Linear(gnn_dim, gnn_dim))
        self.linear_query = nn.Linear(gnn_dim, self.head_count*self.dim_per_head)
        self.linear_key = nn.Linear(2*gnn_dim, self.head_count*self.dim_per_head)
        self.linear_msg = nn.Linear(2*gnn_dim, self.head_count*self.dim_per_head)
        self.mlp = nn.Sequential(nn.Linear(gnn_dim, gnn_dim), nn.BatchNorm1d(gnn_dim), nn.ReLU(), nn.Linear(gnn_dim, gnn_dim))
    
    def forward(self, x, edge_index, edge_type):
        edge_vec = make_one_hot(edge_type, self.n_etype+1)
        loop_edge_vec = torch.zeros(x.size(0), self.n_etype+1).to(edge_vec.device)
        loop_edge_vec[:, self.n_etype] = 1
        edge_vec = torch.cat([edge_vec, loop_edge_vec], dim=0)
        edge_emb = self.edge_encoder(edge_vec)

        loop_index = torch.arange(0, x.size(0), dtype=torch.long, device=edge_index.device)
        loop_index = loop_index.unsqueeze(0).repeat(2,1)
        edge_index = torch.cat([edge_index, loop_index], dim=1)

        aggr_out = self.propagate(edge_index, x=(x,x), edge_attr= edge_emb)
        out = self.mlp(aggr_out)
        return out

    def message(self, edge_index, x_i, x_j, edge_attr):
        # x_i: target, x_j: source #
        query = self.linear_query(x_j).view(-1, self.head_count, self.dim_per_head)
        key = self.linear_key(torch.cat([x_i, edge_attr], dim=1)).view(-1, self.head_count, self.dim_per_head)
        msg = self.linear_msg(torch.cat([x_j, edge_attr], dim=1)).view(-1, self.head_count, self.dim_per_head)

        scores = (query * key).sum(dim=2) / math.sqrt(self.dim_per_head)
        src_node_index = edge_index[0]
        alpha = softmax(scores, src_node_index)

        E = edge_index.size(1)
        N = x_i.size(0)
        ones = torch.full((E,), 1.0, dtype=torch.float).to(edge_index.device)
        src_node_degree = scatter(ones, src_node_index, dim=0, dim_size=N, reduce="sum")[src_node_index]
        alpha = alpha * src_node_degree.unsqueeze(1)

        out = msg * alpha.view(-1, self.head_count, 1)
        return out.view(-1, self.head_count*self.dim_per_head)

def train(args):
    global vocab2vec

    # make initialized weight of each model replica same
    seed_torch(args.seed)

    vocab2vec = pickle.load(open(args.vocab2vec_path, "rb"))

    train_textdir = os.path.join(args.txtdir, "train")
    train_dataset = BERTGNN_Dataset(train_textdir, vocab2vec)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, collate_fn=collate_fn)

    val_textdir = os.path.join(args.txtdir, "valid")
    val_dataset = BERTGNN_Dataset(val_textdir, vocab2vec)
    val_dataloader = DataLoader(val_dataset, batch_size=args.train_batch_size, collate_fn=collate_fn)

    test_textdir = os.path.join(args.txtdir, "test")
    test_dataset = BERTGNN_Dataset(test_textdir, vocab2vec)
    test_dataloader = DataLoader(test_dataset, batch_size=args.train_batch_size, collate_fn=collate_fn)

    model = BERTGNN(args.num_gnn_layer, args.num_edge_type, args.input_dim, args.gnn_dim)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.8)

    result = {}
    result["best_f1"] = 0.0
    result["epoch"] = -1

    start = time.time()
    for epoch in range(args.num_train_epoch):
        model.train()
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            token_vecs = batch["token_vecs"]
            edge_index = batch["edge_index"]
            edge_type = batch["edge_type"]
            label_ids = batch["label_ids"]

            loss = model.log_likelihood(token_vecs, edge_index, edge_type, label_ids)
            loss.backward()
            optimizer.step()

            if i % log_fre == 0:
                print("EPOCH: {} Step: {} Loss: {}".format(epoch, i, loss.data))

        scheduler.step()

        predict(epoch, model, val_dataloader, mode="val", res=result)

def predict(epoch, model, dataloader, mode="val", res=None):
    model.eval()
    with torch.no_grad():
        file_path = predict_file.format(mode, epoch)
        with open(file_path, "w", encoding="utf-8") as fw:
            for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Predicting"):
                b_tokens = batch["b_tokens"]
                token_vecs = batch["token_vecs"]
                edge_index = batch["edge_index"]
                edge_type = batch["edge_type"]
                label_ids = batch["label_ids"]
                pred = model(token_vecs, edge_index, edge_type)
                pred = pred[0]

                for pos, (id_pred, id_ground) in enumerate(zip(pred, label_ids)):
                    if id_ground == tag2idx["PAD"] or id_ground == tag2idx["X"] or id_ground == tag2idx["CLS"] or id_ground == tag2idx["SEP"]:
                        continue
                    else:
                        tag_pred = idx2tag[id_pred] if idx2tag[id_pred] not in ["PAD", "X", "CLS", "SEP"] else "O"
                        tag_ground = idx2tag[id_ground.data.item()]
                        line = "{}\t{}\t{}\n".format(b_tokens[pos], tag_pred, tag_ground)
                        fw.write(line)

        print("=============={} -> {} epoch eval done=================".format(mode, epoch))

        cur_f1 = evaluate_pred_file(file_path)
        if mode == "val":
            if res["best_f1"] < cur_f1:
                res["best_f1"] = cur_f1
                res["epoch"] = epoch
            print("current best f1: {}, epoch: {}".format(res["best_f1"], res["epoch"]))

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
    parser.add_argument("--vocab2vec_path", 
                        default="./data/twitter2015/vocab2vec_dict.pickle", 
                        type=str, 
                        help="Path to the embeddings of twitter2015 tokens")
    parser.add_argument('--ckpt_path', 
                        type=str, 
                        default="./model.pt", 
                        help="path to save or load model")
    parser.add_argument("--num_train_epoch",
                        default=30,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--train_batch_size",
                        default=1,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--test_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--lr",
                        default=0.00005,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--seed',
                        type=int,
                        default=2021,
                        help="random seed for initialization")
    parser.add_argument("--num_gnn_layer",
                        default=5,
                        type=int,
                        help="The number of GNN layers")
    parser.add_argument("--num_edge_type",
                        default=47,
                        type=int,
                        help="The number of edge types")
    parser.add_argument("--input_dim",
                        default=1024,
                        type=int, 
                        help="Dimensions of the input token embeddings")
    parser.add_argument("--gnn_dim", 
                        default=100, 
                        type=int, 
                        help="Dimensions of GNN layers")
    args = parser.parse_args()
    if args.do_train:
        train(args)
    elif args.do_test:
        test(args)
    else:
        raise ValueError('At least one of `do_train`, `do_eval` must be True.')

if __name__ == "__main__":
    main()