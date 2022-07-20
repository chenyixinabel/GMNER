import os, glob, pickle
import spacy
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

nlp = spacy.load("en_core_web_sm")

dep_relations_all = [
    'ROOT',
    'acl',
    'acomp',
    'advcl',
    'advmod',
    'agent',
    'amod',
    'appos',
    'attr',
    'aux',
    'auxpass',
    'case',
    'cc',
    'ccomp',
    'compound',
    'conj',
    'csubj',
    'csubjpass',
    'dative',
    'dep',
    'det',
    'dobj',
    'expl',
    'intj',
    'mark',
    'meta',
    'neg',
    'nmod',
    'npadvmod',
    'nsubj',
    'nsubjpass',
    'nummod',
    'oprd',
    'parataxis',
    'pcomp',
    'pobj',
    'poss',
    'preconj',
    'predet',
    'prep',
    'prt',
    'punct',
    'quantmod',
    'relcl',
    'xcomp'
]

def get_text_file_paths(text_dir):
    text_file_paths = sorted(glob.glob(os.path.join(text_dir, "*_s.txt")), key=lambda x: int(
        x.split('/')[-1].split('_')[0]))  # sort text file names numerically
    return text_file_paths

def extract_dep_relations(doc):
    dep_relations_all2id = {rel: i for i, rel in enumerate(dep_relations_all)}
    dep_relations = []
    for token in doc:
        for child in token.children:
            dep_relations.append(
                (child.i, token.i, dep_relations_all2id[child.dep_]))
    return dep_relations


def extract_seq_relations(doc):
    adj_rel_id = len(dep_relations_all)  # e.g., token a -> token b
    adj_rel_rev_id = adj_rel_id + 1  # e.g., token b -> token a
    seq_relations = [(i, i+1, adj_rel_id)
                     for i in range(len(doc)) if i+1 < len(doc)]
    seq_relations.extend([(i+1, i, adj_rel_rev_id) for i in range(len(doc)) if i+1 < len(doc)])
    return seq_relations

# def get_token_vecs(doc, model, tokenizer):
#     token_vecs = []
#     for token in doc:
#         marked_text = "[CLS] " + token.text + " [SEP]"
#         tokenized_text = tokenizer.tokenize(marked_text)
#         indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
#         segment_ids = [1] * len(indexed_tokens)
#         token_tensor = torch.tensor([indexed_tokens])
#         segment_tensor = torch.tensor([segment_ids])
        
#         with torch.no_grad():
#             output = model(token_tensor, segment_tensor)
#             token_vec = model.pooler(output.last_hidden_state)
#         token_vecs.append(token_vec)

#     token_vecs = torch.cat(token_vecs, dim=0)
#     return token_vecs

def tweet_preprocess(sent, vocab2vec):
    doc = nlp(sent)
    edges = extract_dep_relations(doc) + extract_seq_relations(doc)
    
    tokens = [token.text for token in doc]
    token_vecs = [vocab2vec[token.lower_] for token in doc]
    token_vecs = torch.cat(token_vecs, dim=0)       

    edge_type = torch.zeros(len(edges), dtype=torch.long)
    edge_index = torch.zeros((2, len(edges)), dtype=torch.long)
    for i, (head, tail, rel) in enumerate(edges):
        edge_type[i] = rel
        edge_index[0, i] = head
        edge_index[1, i] = tail

    return tokens, token_vecs, edge_index, edge_type

def label_preprocess(words, sent, labels):
    doc = nlp(sent)
    if len(doc) > len(words): # Some words are splitted into more than one tokens by SpaCy tokenizer
        idx = 0
        idx_ext = 0
        anchor = ""
        labels_ext = []
        # Scan words with idx, and tokens in doc with idx_ext #
        while idx_ext < len(doc):
            # Found a match between words and doc, increase both idx and idx_ext #
            if doc[idx_ext].text == words[idx]:
                labels_ext.append(labels[idx])
                idx += 1
            else:
                anchor += doc[idx_ext].text # Concatenate the token with anchor
                # This token is the first part of the corresponding word #
                if anchor == doc[idx_ext].text:
                    labels_ext.append(labels[idx])
                # This token is the last part of the corresponding word #
                elif anchor == words[idx]:
                    labels_ext.append("X")
                    idx += 1
                    anchor = ""
                else:
                    labels_ext.append("X")
            idx_ext += 1
        return labels_ext
    else:
        return labels

def construct_vocab_part(text_files, token_seen, token_vocab):
    for file in text_files:
        with open(file, "r", encoding="utf-8") as fin:
            l = fin.readline().split("\t")
            doc = nlp(" ".join(l))

            for token in doc:
                if token.lower_ not in token_seen:
                    token_seen.add(token.lower_)
                    token_vocab.append(token.lower_)
    return token_seen, token_vocab

def construct_vocab(text_dir):
    train_text_dir = os.path.join(text_dir, "train")
    val_text_dir = os.path.join(text_dir, "valid")
    test_text_dir = os.path.join(text_dir, "test")

    token_seen = set()
    token_vocab = []

    train_text_files = get_text_file_paths(train_text_dir)
    token_seen, token_vocab = construct_vocab_part(
        train_text_files, token_seen, token_vocab)
    val_text_files = get_text_file_paths(val_text_dir)
    token_seen, token_vocab = construct_vocab_part(
        val_text_files, token_seen, token_vocab)
    test_text_files = get_text_file_paths(test_text_dir)
    token_seen, token_vocab = construct_vocab_part(
        test_text_files, token_seen, token_vocab)

    return token_vocab

def construct_vocab2vec_dict(text_dir, output_path):
    model = BertModel.from_pretrained(
        'bert-large-uncased', output_hidden_states=True)
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

    token_vocab = construct_vocab(text_dir)

    token2vec_dict = {}
    for text in tqdm(token_vocab):
        marked_text = "[CLS] " + text + " [SEP]"
        tokenized_text = tokenizer.tokenize(marked_text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segment_ids = [1] * len(indexed_tokens)
        token_tensor = torch.tensor([indexed_tokens])
        segment_tensor = torch.tensor([segment_ids])
        
        with torch.no_grad():
            output = model(token_tensor, segment_tensor)
            token_vec = model.pooler(output.last_hidden_state)
        token2vec_dict[text] = token_vec

    pickle.dump(token2vec_dict, open(output_path, "wb"))

if __name__ == "__main__":
    dataset_dir = "./data/twitter2015"
    construct_vocab2vec_dict(dataset_dir, "./data/twitter2015/vocab2vec_dict.pickle")