# Wikipedia Link: 'https://en.wikipedia.org/wiki/'+ var = German_Township,_Vanderburgh_County,_Indiana

#gensim==3.0.0 ijson nltk networkx pandas matplotlib SPARQLWrapper rdflib tensorflow_gpu==1.15.0 spacy==2.0.0
# nohup python -u triples.py > nohup_2.out

import requests
import re
import pandas as pd
import os
import logging
import json
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import csv
from SPARQLWrapper import SPARQLWrapper, JSON
from joblib import Parallel, delayed

sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

# time_query = 0
#retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])

retry_strategy = Retry(
    total=50,
    status_forcelist=[429, 500, 502, 503, 504],
    method_whitelist=["HEAD", "GET", "OPTIONS"],
    backoff_factor=15
)
adapter = HTTPAdapter(max_retries=retry_strategy)
http = requests.Session()
http.mount("https://", adapter)
# http.mount("http://", adapter)

# from retry_requests import retry

# SELECT ?rel ?item ?rel2 ?to_item {
#   wd:Q64 ?rel ?item .
#   OPTIONAL {?item ?rel2 ?to_item . } .
#   #FILTER regex (str(?item), 'ˆ((?!statement).)*$') .
#   #FILTER regex (str(?item), 'ˆ((?!https).)*$') .
# } LIMIT 1500



_query_II = '''
SELECT  *
WHERE {
        wd:%s rdfs:label ?label .
        FILTER (langMatches( lang(?label), "EN" ) )
      }
LIMIT 1
'''

class WikidataItems:
    _filename = os.path.join('data/wikidata_items.csv')
    _logger = logging.getLogger(__name__)

    def __init__(self):
        self._logger.warning('Loading items')
        self._items_dict = {}
        self._reverse_dict = {}
        #with open(self._filename, encoding='utf8') as f:
        #    for item in f.readlines():
        #        item = item[:-1]
        #        item_key, item_value = item.split('\t')[:2]
        #        if ':' in item_value or len(item_value) < 2:
        #            continue
        #        if item_key not in self._items_dict:
        #            self._items_dict[item_key] = item_value
        #        try:
        #            self._reverse_dict[item_value.lower()].append(item_key)
        #        except:
        #            self._reverse_dict[item_value.lower()] = [item_key]
        #        # add also string without '.'
        #        try:
        #            self._reverse_dict[item_value.lower().replace('.', '')].append(item_key)
        #        except:
        #            self._reverse_dict[item_value.lower().replace('.', '')] = [item_key]
#
        #        # print(self._items_dict.keys())
        self._logger.warning('Items loaded')

    def __getitem__(self, item):
        return self._items_dict[item]

    def translate_from_url(self, url):
        if '/' in url and '-' not in url:
            item = url.split('/')[-1]
        elif '/' in url and '-' in url:
            item = url.split('/')[-1].split('-')[0]
        else:
            item = url
        try:
            result = self._items_dict[item]
        except:
            # url = 'https://query.wikidata.org/bigdata/namespace/wdq/sparql'
            # # url = "http://131.220.9.218/wikidatadbp2k18/sparql"
            # data = requests.get(url, params={'query': _query_II % item,
            #                                  'format': 'json'}) #.json()
            # if data.status_code != 200:
            #     time.sleep(3)
            #     data = requests.get(url, params={'query': _query % item,
            #                                      'format': 'json'})
            try:
                t1 = time.time()
                sparql.setQuery(_query_II % item)
                sparql.setReturnFormat(JSON)
                data = sparql.query().convert()
                # time_query += time.time() - t1
            except:
                t2 = time.time()
                sparql.setQuery(_query % item)
                sparql.setReturnFormat(JSON)
                data = sparql.query().convert()
                # time_query += time.time() - t2
            try:
                # data = data.json()
                result = data['results']['bindings'][0]['label']['value']
            except:
                result = ''
        return result

    def reverse_lookup(self, word):
        return self._reverse_dict[word.lower()]

wikidata_items = WikidataItems()


query_nn_back = '''
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?item0 ?rel ?item1 WHERE {
  ?item0 rdfs:label "%s"@en .
  ?item1 ?rel ?item0 .
  FILTER regex (str(?item1), '^((?!statement).)*$') .
  FILTER regex (str(?item1), '^((?!https).)*$') .
} LIMIT 1500
'''

query_nn2_back = '''
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?item0 ?rel ?item1 WHERE {
  ?item rdfs:label "%s"@en .
  ?item0 ?r ?item .
  ?item1 ?rel ?item0 .
  FILTER regex (str(?item0), '^((?!statement).)*$') .
  FILTER regex (str(?item0), '^((?!https).)*$') .
  FILTER regex (str(?item1), '^((?!statement).)*$') .
  FILTER regex (str(?item1), '^((?!https).)*$') .
} limit 1000
'''

query_nn3_back = '''
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?item0 ?rel ?item1 WHERE {
  ?item rdfs:label "%s"@en .
  ?item_nn ?r ?item .
  ?item0 ?r2 ?item_nn .
  ?item1 ?rel ?item0 .
} limit 1000
'''

query_nn_forw = '''
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?item0 ?rel ?item1  WHERE {
  ?item0 rdfs:label "%s"@en .
  ?item0 ?rel ?item1 .
  FILTER regex (str(?item1), '^((?!statement).)*$') .
  FILTER regex (str(?item1), '^((?!https).)*$') .
} limit 5000
'''

query_nn2_forw = '''
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?item0 ?rel ?item1 WHERE {
  ?item rdfs:label "%s"@en .
  ?item ?r ?item0 .
  ?item0 ?rel ?item1 .
  FILTER regex (str(?item0), '(statement)') .  # two hops only for in-line statements
  FILTER regex (str(?item1), '^((?!statement).)*$') .
  FILTER regex (str(?item1), '^((?!https).)*$') .
} limit 1000
'''

query_nn3_forw = '''
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?item0 ?rel ?item1 WHERE {
  ?item rdfs:label "%s"@en .
  ?item ?r ?item_nn .
  ?item_nn ?r2 ?item0 .
  ?item0 ?rel ?item1 .
} limit 1000
'''


# ADDED
_query_hop_II = '''
SELECT ?rel ?item ?rel2 ?to_item {
  wd:%s ?rel ?item
  OPTIONAL { ?item ?rel2 ?to_item }
  FILTER regex (str(?item), '^((?!statement).)*$') .
  FILTER regex (str(?item), '^((?!https).)*$') .
} LIMIT 1500
'''

_query = '''
SELECT ?rel ?item {
  wd:%s ?rel ?item
  FILTER regex (str(?item), '^((?!statement).)*$') .
  FILTER regex (str(?item), '^((?!https).)*$') .

  FILTER(!isLiteral(?item) || langMatches(lang(?item), "en"))

} LIMIT 50
'''

#LIMIT 100
##SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
# {"batchcomplete":"",
#  "query":{"normalized":[{"from":"Kofoworola_Abeni_Pratt","to":"Kofoworola Abeni Pratt"}],
#           "pages":{"51046741":{"pageid":51046741,"ns":0,"title":"Kofoworola Abeni Pratt",
#           "pageprops":{"defaultsort":"Pratt, Kofoworola Abeni","wikibase-shortdesc":"20th-century Nigerian-born nurse;
#           first black Chief Nursing Officer of Nigeria","wikibase_item":"Q25796287"}}}}}

def get_wikidata_id_from_wikipedia_id(wikipedia_id):
    url = "https://en.wikipedia.org/w/api.php?action=query&prop=pageprops&titles="+wikipedia_id+"&format=json"
    # url = 'https://en.wikipedia.org/w/api.php?action=query&prop=pageprops&pageids=%s&format=json' % str(
    #     wikipedia_id)
    try:
        res = requests.get(url).json()['query']['pages']
        pid = list(res.keys())[0]
        return res[pid]['pageprops']['wikibase_item']
    except:
        return ''

# def get_graph_from_wikidata_id(wikidata_id, central_item):
#     url = 'https://query.wikidata.org/bigdata/namespace/wdq/sparql'
#     data = requests.get(url, params={'query': _query % wikidata_id,'format': 'json'})
#     if data.status_code != 200:
#         time.sleep(40)
#         data = requests.get(url, params={'query': _query % wikidata_id,'format': 'json'})
#         if data.status_code != 200:
#             time.sleep(160)
#             data = requests.get(url, params={'query': _query % wikidata_id,'format': 'json'})
#     data = data.json()
#     print("Reading")
#     sys.stdout.flush()
#     triplets = ""
#     graph_set = set()
#     for item in data['results']['bindings']:
#         try:
#             from_item = wikidata_items.translate_from_url(wikidata_id)
#             relation = wikidata_items.translate_from_url(item['rel']['value'])
#             to_item = wikidata_items.translate_from_url(item['item']['value'])
#             graph_set.add(relation)
#             graph_set.add(to_item)
#         except Exception as e:
#             pass
#         """try:
#             from_item = wikidata_items.translate_from_url(item['item']['value'])
#             relation = wikidata_items.translate_from_url(item['rel2']['value'])
#             to_item = wikidata_items.translate_from_url(item['to_item']['value'])
#             #triplets.append((from_item, relation, to_item))
#             graph_set.add(relation)
#             graph_set.add(to_item)
#         except Exception as e:
#             #print("Here ", e)
#             pass"""
#             #sys.exit()
#     #triplets = sorted(list(set(triplets)))
#     triplets = ' '.join(graph_set)
#     if triplets == "":
#         raise RuntimeError("This graph contains no suitable triplets.")
#     return triplets

def get_graph_from_wikidata_id(wikidata_id):
    # url = 'https://query.wikidata.org/bigdata/namespace/wdq/sparql'
    # # url = "http://131.220.9.218/wikidatadbp2k18/sparql"
    # # response = http.get("https://en.wikipedia.org/w/api.php")

    # data = http.get(url, params={'query': _query % wikidata_id,
    #                                  'format': 'json'})  # .json()
    # print("data >> ",data)
    # if data.status_code != 200:
    #     time.sleep(1.5)
    #     data = requests.get(url, params={'query': _query % wikidata_id,
    #                                      'format': 'json'}).json()
    #     print("data >> ",data)
    #     if data.status_code != 200:
    #         time.sleep(3)
    #         data = requests.get(url, params={'query': _query % wikidata_id,
    #                                          'format': 'json'}).json()
    #         print("data >> ",data)
    t3 = time.time()
    sparql.setQuery(_query % wikidata_id)
    sparql.setReturnFormat(JSON)
    data = sparql.query().convert()
    # time_query += time.time() - t3
    try :
        # data = data.json()

        # triplets = []
        print(len(data['results']['bindings']))
        # for item in data['results']['bindings']:
        #     print("item >>> ",item)
        #     #if 'datatype' in item['item'].keys():
        #     #    continue
        #     #elif item['item']['type'] == 'literal' and (bool(re.match('[\d/_\W]+$', item['item']['value'])) or
        #     #                                             bool(re.match('.[\d/_\W]+$', item['item']['value'])) or
        #     #                                             bool(re.match('[\d/_\W]+.$', item['item']['value']))):
        #     #    continue
        #     #elif 'xml:lang' in item['item'].keys() and item['item']['xml:lang'] != 'en':
        #     #    continue

        #     triplets.append(fun(wikidata_id, item))
        triplets = Parallel(n_jobs=-1)(delayed(fun)(wikidata_id, item) for item in data['results']['bindings'])
    except:
        pass

    return triplets

def fun(wikidata_id, item):
    from_item = wikidata_items.translate_from_url(wikidata_id)
    relation = item['rel']['value']

    required_dic = {'core#altLabel':'also known as','description':'description','rdf-schema#label':'label'}
    if relation.split('/')[-1] in required_dic.keys() :
        relation = required_dic[relation.split('/')[-1]]
    else:
        relation = wikidata_items.translate_from_url(item['rel']['value'])
    to_item = item['item']['value']
    #if relation == 'Freebase ID':
    #    continue
    if to_item.startswith('http'):
        to_item = wikidata_items.translate_from_url(to_item)
    if to_item == '': #result
        return ""
    return from_item+","+relation+","+to_item 

def get_triplets_for_word_2_hops(word):
    # http://131.220.9.218/wikidatadbp2k18/sparql
    # url = 'https://query.wikidata.org/bigdata/namespace/wdq/sparql'
    url = "http://131.220.9.218/wikidatadbp2k18/sparql"
    triplets = []
    for query in [query_nn_forw, query_nn2_forw]:
        data = requests.get(url, params={'query': query % word,
                                         'format': 'json'}).json()
        if data.status_code != 200:
            time.sleep(5)
            data = requests.get(url, params={'query': query % word,
                                            'format': 'json'}).json()
            if data.status_code != 200:
                time.sleep(10)
                data = requests.get(url, params={'query': query % word,
                                                'format': 'json'}).json()
        for item in data['results']['bindings']:
            try:
                if 'datatype' in item['item1'].keys():
                    continue
                elif item['item1']['type'] == 'literal' and (bool(re.match('[\d/_\W]+$', item['item1']['value'])) or
                                                             bool(re.match('.[\d/_\W]+$', item['item1']['value'])) or
                                                             bool(re.match('[\d/_\W]+.$', item['item1']['value']))):
                    continue
                elif ('xml:lang' in item['item1'].keys()) and not (item['item1']['xml:lang'] == 'en'):
                    continue
                elif item['item1']['type'] == 'uri':
                    pass
                # to_item = wikidata_items.translate_from_url(item['item1']['value']) # + '|' + item['item1']['value']
                # relation = wikidata_items.translate_from_url(item['rel']['value']) # + '|' + item['rel']['value']
                # from_item = wikidata_items.translate_from_url(item['item0']['value'])#  + '|' + item['item0']['value']
                # triplets.append((from_item, relation, to_item))
            except:
                pass
    return triplets


def get_triplets_for_word_1_hop(word):
    url = 'https://query.wikidata.org/bigdata/namespace/wdq/sparql'
    triplets = []
    id_rels = ['http://www.wikidata.org/prop/direct/P356',]
    prev_val = ''
    for query in [query_nn_forw]:
        data = requests.get(url, params={'query': query % word,
                                         'format': 'json'}).json()

        for item in data['results']['bindings']:
            try:
                # 1. if item['item1']['xml:lang'] == 'en' :
                if 'datatype' in item['item1'].keys():
                    continue
                elif item['item1']['type']=='literal' and ( bool(re.match('[\d/_\W]+$', item['item1']['value'])) or
                                                        bool(re.match('.[\d/_\W]+$', item['item1']['value'])) or
                                                            bool(re.match('[\d/_\W]+.$', item['item1']['value']))):
                         continue
                elif ('xml:lang' in item['item1'].keys()) and not (item['item1']['xml:lang'] == 'en') :
                    continue
                elif item['item1']['type'] == 'uri':
                    pass
                    #print('X')
                    #print(item)
                    #print(item['item1']['xml:lang'])
                    # to_item = wikidata_items.translate_from_url(item['item1']['value']) # + '|' + item['item1']['value']
                    # relation = wikidata_items.translate_from_url(item['rel']['value']) # + '|' + item['rel']['value']
                    # from_item = wikidata_items.translate_from_url(item['item0']['value']) # + '|' + item['item0']['value']
                    # triplets.append((from_item, relation, to_item))
                else:
                    print(item)
            except:
                print(item)
            #     pass

    return triplets


if __name__ == '__main__':
    start_time = time.time()
    print('Getting triplets')
    triplets = get_graph_from_wikidata_id('Q30055')
    print("TRIPLES >>> ",triplets)
    print("--- %s seconds ---" % (time.time() - start_time))
    # print("--- %s seconds ---" % (time_query))
    # triplets = get_triplets_for_word_1_hop('Brazil')
    # [print(triplet) for triplet in triplets]
    # print(len(triplets))
    # triplets = get_graph_from_wikidata_id('Q155')
    # [print(triplet) for triplet in triplets]
    # print(len(triplets))

    # Load the dataset
    # dataset_file = "data/aida_train.csv"
    # dataset = pd.read_csv(dataset_file, sep='\t', lineterminator='\n')
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)
    # print(dataset.head(1))

    # dataset_file = "data/ent2desc.json"
    # with open(dataset_file, "r") as read_file:
    #     dataset = json.load(read_file)
    #
    #     keys = list(dataset.keys())
    #
    #     out_f = open('keys.txt','w+')
    #     out_f.write('\t'.join(keys)+'\n')
    #     out_f.close()
    #
    #     values = list(dataset.values())
    """
    keys_file = 'data/myfile.txt'
    with open(keys_file, "r") as read_file:
        keys = []
        for row in read_file:
            keys.extend(row.strip().split('\t'))

        total_items = len(keys)
        print("All counts : ",total_items)


        max = 80
        end_first = max//3
        start_second = end_first
        end_second = end_first * 2
        start_third = end_second
        iterator = 0 #Start of sixth 6th :: This is where Server 04 should end at ###end

        # for key, value in dataset.items():
        #     wikidata_id = get_wikidata_id_from_wikipedia_id(key)
        #     triplets = get_graph_from_wikidata_id(wikidata_id)
        #     print(wikidata_id + " :  ",triplets,"\n")
        #
        #     iterator += 1

        header = "WikipediaID\tWikidataID\ttriples"
        full_data = []
        missing_wiki_ids = []

        while iterator < max:
            key = keys[iterator]
            wikidata_id = get_wikidata_id_from_wikipedia_id(key)
            triplets = get_graph_from_wikidata_id(wikidata_id)
            full_data.append(key+"\t"+wikidata_id+"\t"+" || ".join(triplets))
            print(str(iterator)+"\t"+key+"\t"+wikidata_id+"\t"+" || ".join(triplets)+"\n")
            iterator += 1

        # Output Triples
        out_file = "triples_part_3.tsv"
        with open(out_file,'a+') as  f:
            f.write(header+'\n')

            for data_point in full_data :
                f.write(data_point + '\n')
        f.close()
    """
