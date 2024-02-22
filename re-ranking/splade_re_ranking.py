#!/usr/bin/env python3
# Load a patched ir_datasets that loads the injected data inside the TIRA sandbox
from tira.third_party_integrations import load_rerank_data, persist_and_normalize_run, ensure_pyterrier_is_loaded, get_output_directory
from pathlib import Path
import pandas as pd

def get_splade():
    ensure_pyterrier_is_loaded()
    import pyt_splade
    return pyt_splade.SpladeFactory("/workspace/splade-cocondenser-ensembledistil")

def process(text, splade):
    ret = [{'docno': "1", 'text': text}]
    ret = splade.indexing()(ret)
    del ret['text']
    return ret.iloc[0].to_dict()["toks"]

def score_query_document_pair(query_bow, doc_bow):
    sc = 0
    for id_, v in query_bow.items():
        if id_ in doc_bow:
            sc += v * doc_bow[id_]
    return sc

if __name__ == '__main__':
    # In the TIRA sandbox, this is the injected re-ranking dataset, injected via the environment variable TIRA_INPUT_DIRECTORY
    re_rank_dataset = load_rerank_data(default='workshop-on-open-web-search/re-ranking-20231027-training')

    ensure_pyterrier_is_loaded()
    print('Step 1: Load splade.')
    splade = get_splade()

    q_id_to_bow = {}
    d_id_to_bow = {}

    # Alternatively, you could use the scored docs of ir_datasets, e.g.:
    # from tira.third_party_integrations import ir_dataset
    # dataset = ir_datasets.load(default='workshop-on-open-web-search/document-processing-20231027-training')

    run = []

    for _, i in re_rank_dataset.iterrows():

        if i['qid'] not in q_id_to_bow:
            q_id_to_bow[i['qid']] = process(i['query'], splade)
        if i['docno'] not in d_id_to_bow:
            d_id_to_bow[i['docno']] = process(i['text'], splade)

        run += [{'qid': i['qid'], 'docno': i['docno'], 'score': score_query_document_pair(q_id_to_bow[i["qid"]], d_id_to_bow[i["docno"]])}]
    
    run = pd.DataFrame(run)
    
    output_dir = get_output_directory('.')

    docs_df = pd.DataFrame([{'docno': i, 'toks': d_id_to_bow[i]} for i in d_id_to_bow])
    docs_df.to_json(f'{output_dir}/documents.jsonl.gz', lines=True, orient='records')

    queries_df = pd.DataFrame([{'qid': i, 'toks': q_id_to_bow[i]} for i in q_id_to_bow])
    queries_df.to_json(f'{output_dir}/queries.jsonl.gz', lines=True, orient='records')

    # Re-rankers are expected to produce a TREC-style run.txt file in the output directory.
    persist_and_normalize_run(run, 'my-system-name', default_output=f'{output_dir}/run.txt')
    
    # You can pass additional arguments to your program, e.g., via argparse, to modify the behaviour of your program.
    
