import pickle
import os

mode = "train"

entity_catalogue_path = "wikilinks_entity_catalogue.jsonl"
data_path = "data/wikilinks/%s.jsonl" % mode

with open(data_path, "r") as fin:
    samples = fin.readlines()

with open(entity_catalogue_path, "r") as fin:
    entity_catalogue = fin.readlines()

f = open("model/wikilinks/top_16_candidates/correct_pred_mentions_indices_%s.pickle" % mode, "wb")
correct_pred_mention_indices = pickle.load(f)

f = open("model/wikilinks/top_16_candidates/%s_indices.pickle" % mode, "wb")
candidate_indices = pickle.load(f)

fname = os.path.join("model/wikilinks/top_16_candidates", "%s.t7" % mode)
biencoder_data = torch.load(fname)
labels = biencoder_data["labels"]

correct_samples = [samples[i] for i in correct_pred_mention_indices]

for i in range(len(correct_samples)):
    sample = correct_samples[i]
    d = {}
    d['ctxt_left'] = sample['context_left']
    d['mention'] = sample['mention']
    d['ctxt_right'] = sample['context_right']
    d['label_idx'] = labels[i]
    candidates = candidate_indices[i]
    topk_attributes = []
    for cand in candidates:
        entity_attributes = {}
        entity = entity_catalogue[cand]
        entity_attributes['QID'] = entity[]
        entity_attributes['WIKIPEDIA_ID'] = entity['idx'][entity['idx'].find('?curid=')+7:]
        entity_attributes['entity'] = entity['entity']
        entity_attributes['title'] = entity['title']
        entity_attributes['wikipedia_desc'] = entity['text']
        entity_attributes['wikidata_desc'] = entity['wikidata_desc']
        entity_attributes['wikidata_alias'] = entity['wikidata_alias']
        entity_attributes['wikidata_instanceof'] = entity['wikidata_instance']

        topk_attributes.append(entity_attributes)

    d['topk'] = topk_attributes
