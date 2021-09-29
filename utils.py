#data loading
#data preprocessing
import json
import pickle
from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch.nn as nn

class RobertaForMaskedLMwithLoss(RobertaForMaskedLM):
    #
    def __init__(self, config):
        super().__init__(config)
    #
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, masked_lm_labels=None):
        #
        assert attention_mask is not None
        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask)
        sequence_output = outputs[0] #hidden_states of final layer (batch_size, sequence_length, hidden_size)
        prediction_scores = self.lm_head(sequence_output)
        outputs = (prediction_scores, sequence_output) + outputs[2:]
        if masked_lm_labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            bsize, seqlen = input_ids.size()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1)).view(bsize, seqlen)
            masked_lm_loss = (masked_lm_loss * attention_mask).sum(dim=1)
            outputs = (masked_lm_loss,) + outputs
            # (masked_lm_loss), prediction_scores, sequence_output, (hidden_states), (attentions)
        return outputs

TOKENIZER = RobertaTokenizer.from_pretrained('roberta-large')
LM_MODEL = RobertaForMaskedLMwithLoss.from_pretrained('roberta-large')
LM_MODEL.cuda(); LM_MODEL.eval()

def generate_node_score_with_LM(eids, mention, id2nodeText):
    # returns node_score: (max_node_num)

    eids = eids[:]
    eids.insert(0, -1) #QAcontext node
    sents, scores = [], []
    for eid in eids:
        if eid==-1:
            sent = mention.lower()
        else:
            sent = '{} {}.'.format(mention.lower(), id2nodeText[eid])
        sent = TOKENIZER.encode(sent, add_special_tokens=True)
        sents.append(sent)

    n_eids = len(eids)
    cur_idx = 0
    batch_size = 50
    while cur_idx < n_eids:
        #Prepare batch
        input_ids = sents[cur_idx: cur_idx+batch_size]
        max_len = max([len(seq) for seq in input_ids])
        for j, seq in enumerate(input_ids):
            seq += [TOKENIZER.pad_token_id] * (max_len-len(seq))
            input_ids[j] = seq
        input_ids = torch.tensor(input_ids).cuda() #[B, seqlen]
        mask = (input_ids!=1).long() #[B, seq_len]
        #Get LM score
        with torch.no_grad():
            outputs = LM_MODEL(input_ids, attention_mask=mask, masked_lm_labels=input_ids)
            loss = outputs[0] #[B, ]
            _scores = list(-loss.detach().cpu().numpy()) #list of float
        scores += _scores
        cur_idx += batch_size
    assert len(sents) == len(scores) == len(eids)
    eid2score = OrderedDict(sorted(list(zip(eids, scores)), key=lambda x: -x[1])) #score: from high to low
    return eid2score



# {'ctxt_left':<LEFT CONTEXT>, 'mention':<MENTION STRING>, 'ctxt_right':<RIGHT CONTEXT>, 'label_idx':<ID>, 'topk':[{'QID':<QID>, 'WID':<CURID>, 'entity':<ENTITY>, 'title':<TITLE>, 'wikipedia_desc':<WIKIPEDIA DESCRIPTION>, 'wikidata_desc':<WIKIDATA DESCRIPTION>, 'wikidata_alias':[<ALIAS1>, ..., <ALIASn>], 'wikidata_instanceof':[<INSTANCE1>, ..., <INSTANCEn>],},{}...,{}]}
def load_sparse_adj_data_with_contextnode(data_path, max_node_num):

    #Returns node_ids, node_type_ids, node_scores, adj_lengths, (edge_index, edge_type)
    #node_ids: (num_mentions, num_candidates, max_node_num)
    #node_type_ids: (num_mentions, num_candidates, max_node_num)
    #node_scores: (num_mentions, num_candidates, max_node_num)
    #adj_lengths: (num_mentions, num_candidates)
    #edge_index: list of size (num_mentions, num_candidates), where each entry is tensor[2, E]
    #edge_type: list of size (num_mentions, num_candidates), where each entry is tensor[E, ]

    id2nodeText = {0:"<pad>"}
    with open(data_path, "r") as fin:
        lines = fin.readlines()
        n_samples = len(lines)
        n_candidates = len(json.loads(lines[0])['topk'])
        edge_index, edge_type = [], []
        adj_lengths = torch.zeros((n_samples, n_candidates), dtype=torch.long)
        node_ids = torch.full((n_samples, num_candidates, max_node_num), 1, dtype=torch.long)
        node_type_ids = torch.full((n_samples, num_candidates, max_node_num), 2, dtype=torch.long) #0: "mention node", 1: "candidate node", default 2: "other node", 3: context node
        node_scores = torch.zeros((n_samples, num_candidates, max_node_num), dtype=torch.float)
        i = 0; idx = 1
        for line in lines:
            sample = json.loads(line)
            node_ids[i,:,0] = 0
            node_type_ids[i,:,0] = 3
            for j in range(len(sample['topk'])):
                edge_from, edge_to, edge_relation = [], [], []
                k = 1
                if sample['topk'][j]['wikipedia_desc']:
                    node_type_ids[i,j,k] = 1
                    node_ids[i,j,k] = idx
                    id2nodeText[idx] = sample['topk'][j]['wikipedia_desc']
                    edge_from.append(idx)
                    edge_to.append(0)
                    edge_relation.append(0)
                    en_node = idx
                    k += 1
                    idx += 1
                if sample['topk'][j]['wikidata_desc']:
                    node_ids[i,j,k] = idx
                    id2nodeText[idx] = sample['topk'][j]['wikidata_desc']
                    edge_from.append(idx)
                    edge_to.append(en_node)
                    edge_relation.append(1)
                    k += 1
                    idx += 1
                for alias in sample['topk'][j]['wikidata_alias']):
                    node_ids[i,j,k] = idx
                    id2nodeText[idx] = sample['topk'][j]['wikidata_alias']
                    edge_from.append(idx)
                    edge_to.append(en_node)
                    edge_relation.append(2)
                    k += 1
                    idx += 1
                for instanceof in sample['topk'][j]['wikidata_instanceof']):
                    node_ids[i,j,k] = idx
                    id2nodeText[idx] = sample['topk'][j]['wikidata_instanceof']
                    edge_from.append(idx)
                    edge_to.append(en_node)
                    edge_relation.append(3)
                    k += 1
                    idx += 1
                adj_lengths[i,j] = k
                edge_from = torch.tensor(edge_from, dtype=torch.long)
                edge_to = torch.tensor(edge_to, dtype=torch.long)
                edge_relation = torch.tensor(edge_relation, dtype=torch.long)
                edge_from, edge_to, edge_relation = torch.cat((edge_from, edge_to), 0), torch.cat((edge_to, edge_from), 0), torch.cat((edge_relation, edge_relation+4), 0)
                edge_index.append(torch.stack([edge_from,edge_to], dim=0)) #each entry is [2, E]
                edge_type.append(edge_relation) #each entry is [E, ]
                node_scores[i,j] = generate_node_score_with_LM(sample['ctxt_left'] + ' ' + sample['mention'] + ' ' + sample['ctxt_right'], node_ids[i,j], id2nodeText)
            i += 1

        with open('id2nodeText.pickle', 'wb') as handle:
            pickle.dump(id2nodeText, handle, protocol=pickle.HIGHEST_PROTOCOL)

        edge_index = list(map(list, zip(*(iter(edge_index),) * n_candidates)))
        edge_type = list(map(list, zip(*(iter(edge_type),) * n_candidates)))

        return node_ids, node_type_ids, node_scores, adj_lengths, (edge_index, edge_type)

def load_input_tensors(input_jsonl_path, model_type, model_name, max_seq_length):
    if model_type in ('lstm',):
        raise NotImplementedError
    elif model_type in ('gpt',):
        return load_gpt_input_tensors(input_jsonl_path, max_seq_length)
    elif model_type in ('bert', 'xlnet', 'roberta', 'albert'):
        return load_bert_xlnet_roberta_input_tensors(input_jsonl_path, model_type, model_name, max_seq_length)

def load_bert_xlnet_roberta_input_tensors(statement_jsonl_path, model_type, model_name, max_seq_length):
    class InputExample(object):

        def __init__(self, example_id, question, ctxt_left, ctxt_right, mention, endings, label=None):
            self.example_id = example_id
            self.question = question
            self.ctxt_left = ctxt_left
            self.ctxt_right = ctxt_right
            self.mention = mention
            self.endings = endings
            self.label = label

    class InputFeatures(object):

        def __init__(self, example_id, choices_features, label):
            self.example_id = example_id
            self.choices_features = [
                {
                    'input_ids': input_ids,
                    'input_mask': input_mask,
                    'segment_ids': segment_ids,
                    'output_mask': output_mask,
                }
                for _, input_ids, input_mask, segment_ids, output_mask in choices_features
            ]
            self.label = label

    def read_examples(input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            examples = []
            i = 0
            for line in f.readlines():
                sample = json.loads(line)
                label = sample['label_idx']
                ctxt_left = sample['ctxt_left']
                mention = sample['mention']
                ctxt_right = sample['ctxt_right']
                examples.append(
                    InputExample(
                        example_id=i,
                        ctxt_left = [ctxt_left] * len(sample['topk']),
                        mention = [mention] * len(sample['topk']),
                        ctxt_right = [ctxt_right] * len(sample['topk']),
                        question="",
                        endings=[ending["wikipedia desc"] for ending in sample['topk']],
                        label=label
                    ))
                i += 1
        return examples

    def convert_examples_to_features(examples, label_list, max_seq_length,
                                     tokenizer,
                                     cls_token_at_end=False,
                                     cls_token='[CLS]',
                                     cls_token_segment_id=1,
                                     sep_token='[SEP]',
                                     ent_start_token="[unused0]",
                                     ent_end_token="[unused1]",
                                     sequence_a_segment_id=0,
                                     sequence_b_segment_id=1,
                                     sep_token_extra=False,
                                     pad_token_segment_id=0,
                                     pad_on_left=False,
                                     pad_token=0,
                                     mask_padding_with_zero=True):
        """ Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """
        label_map = {label: i for i, label in enumerate(label_list)}

        features = []
        for ex_index, example in enumerate(examples):
            choices_features = []
            for ending_idx, (ctxt_left, mention, ctxt_right, ending) in enumerate(zip(example.ctxt_left, example.mention, example.ctxt_right, example.endings)):
                tokens_a1 = tokenizer.tokenize(ctxt_left)
                tokens_a2 = tokenizer.tokenize(mention)
                tokens_a3 = tokenizer.tokenize(ctxt_right)
                tokens_a = tokens_a1 + ent_start_token + tokens_a2 + ent_end_token + tokens_a3
                tokens_b = tokenizer.tokenize(example.question + " " + ending)

                special_tokens_count = 4 if sep_token_extra else 3
                _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count, ent_start_token, ent_end_token)

                # The convention in BERT is:
                # (a) For sequence pairs:
                #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
                #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
                # (b) For single sequences:
                #  tokens:   [CLS] the dog is hairy . [SEP]
                #  type_ids:   0   0   0   0  0     0   0
                #
                # Where "type_ids" are used to indicate whether this is the first
                # sequence or the second sequence. The embedding vectors for `type=0` and
                # `type=1` were learned during pre-training and are added to the wordpiece
                # embedding vector (and position vector). This is not *strictly* necessary
                # since the [SEP] token unambiguously separates the sequences, but it makes
                # it easier for the model to learn the concept of sequences.
                #
                # For classification tasks, the first vector (corresponding to [CLS]) is
                # used as as the "sentence vector". Note that this only makes sense because
                # the entire model is fine-tuned.
                tokens = tokens_a + [sep_token]
                if sep_token_extra:
                    # roberta uses an extra separator b/w pairs of sentences
                    tokens += [sep_token]

                segment_ids = [sequence_a_segment_id] * len(tokens)

                if tokens_b:
                    tokens += tokens_b + [sep_token]
                    segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

                if cls_token_at_end:
                    tokens = tokens + [cls_token]
                    segment_ids = segment_ids + [cls_token_segment_id]
                else:
                    tokens = [cls_token] + tokens
                    segment_ids = [cls_token_segment_id] + segment_ids

                input_ids = tokenizer.convert_tokens_to_ids(tokens)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.

                input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
                special_token_id = tokenizer.convert_tokens_to_ids([cls_token, sep_token])
                output_mask = [1 if id in special_token_id else 0 for id in input_ids]  # 1 for mask

                # Zero-pad up to the sequence length.
                padding_length = max_seq_length - len(input_ids)
                if pad_on_left:
                    input_ids = ([pad_token] * padding_length) + input_ids
                    input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                    output_mask = ([1] * padding_length) + output_mask

                    segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                else:
                    input_ids = input_ids + ([pad_token] * padding_length)
                    input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                    output_mask = output_mask + ([1] * padding_length)
                    segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

                assert len(input_ids) == max_seq_length
                assert len(output_mask) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                choices_features.append((tokens, input_ids, input_mask, segment_ids, output_mask))
            label = label_map[example.label]
            features.append(InputFeatures(example_id=example.example_id, choices_features=choices_features, label=label))

        return features

    def _truncate_seq_pair(tokens_a, tokens_b, max_length, ent_start_token, ent_end_token):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_b) > 0:
                tokens_b.pop()
            else:
                if tokens_a[-1] != ent_end_token:
                    tokens_a.pop()
                else:
                    del tokens_a[0]

    def select_field(features, field):
        return [[choice[field] for choice in feature.choices_features] for feature in features]

    def convert_features_to_tensors(features):
        all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)
        all_output_mask = torch.tensor(select_field(features, 'output_mask'), dtype=torch.bool)
        all_label = torch.tensor([f.label for f in features], dtype=torch.long)
        return all_input_ids, all_input_mask, all_segment_ids, all_output_mask, all_label

    try:
        tokenizer_class = {'bert': BertTokenizer, 'xlnet': XLNetTokenizer, 'roberta': RobertaTokenizer, 'albert': AlbertTokenizer}.get(model_type)
    except:
        tokenizer_class = {'bert': BertTokenizer, 'xlnet': XLNetTokenizer, 'roberta': RobertaTokenizer}.get(model_type)
    tokenizer = tokenizer_class.from_pretrained(model_name)
    examples = read_examples(statement_jsonl_path)
    features = convert_examples_to_features(examples, list(range(len(examples[0].endings))), max_seq_length, tokenizer,
                                            cls_token_at_end=bool(model_type in ['xlnet']),  # xlnet has a cls token at the end
                                            cls_token=tokenizer.cls_token,
                                            sep_token=tokenizer.sep_token,
                                            sep_token_extra=bool(model_type in ['roberta', 'albert']),
                                            cls_token_segment_id=2 if model_type in ['xlnet'] else 0,
                                            pad_on_left=bool(model_type in ['xlnet']),  # pad on the left for xlnet
                                            pad_token_segment_id=4 if model_type in ['xlnet'] else 0,
                                            sequence_b_segment_id=0 if model_type in ['roberta', 'albert'] else 1)
    example_ids = [f.example_id for f in features]
    *data_tensors, all_label = convert_features_to_tensors(features)
    return (example_ids, all_label, *data_tensors)

#create jsonl by running biencoder
#complete dataloader
