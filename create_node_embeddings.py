import torch.nn.functional as F
import torch.nn as nn
from transformers import RobertaModel
from transformers import RobertaConfig
import pickle
import numpy as np

model_type = "roberta_large"
id2text_path = ""
node_embed_file = ""
roberta_dim = 1024
max_seq_length = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #.to(device)

config = RobertaConfig.from_pretrained(model_type)
tokenizer = AutoTokenizer.from_pretrained(model_type, do_lower_case=True)
model = RobertaModel.from_pretrained(model_type, config=config)
model.to(device)

f = open(id2text_path, "rb")
id2text = pickle.load(f)

num_total_nodes = len(id2text)
node_sents = [id2text[k] for k in id2text.keys()]

batch_size = 8
num_batches = (num_total_nodes//batch_size) if num_total_nodes%batch_size==0 else (num_total_nodes//batch_size) + 1

node_embeds = np.zeros(num_total_nodes, roberta_dim)

for b in num_batches:
    batch_sents = node_sents[b*batch_size : min((b+1)*batch_size, num_total_nodes)]
    batch_sents_tokens = tokenizer(batch_sents, padding="max_length", truncation=True,
                                    max_length=max_seq_length)
    batch_input = {k: torch.LongTensor(v).to(device) for k, v in batch_sents_tokens.items()}
    out = model(**batch_input).last_hidden_state[:, 0, :]
    node_embeds[b*batch_size : min((b+1)*batch_size, num_total_nodes)] = out.detach().cpu().numpy()

f = open(node_embed_file, "wb")
np.save(f, node_embeds)
