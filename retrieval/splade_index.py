import argparse
import pyterrier as pt
import os
from math import floor
from more_itertools import chunked
from tqdm import tqdm
import pandas as pd
from tira.third_party_integrations import ensure_pyterrier_is_loaded, persist_and_normalize_run

def parse_args():
    parser = argparse.ArgumentParser(prog='Index with SPLADE for Pyterrier.')

    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch', default=os.environ.get('BATCH_SIZE', 128), type=int)

    return parser.parse_args()

def get_splade():
    ensure_pyterrier_is_loaded()
    import pyt_splade
    return pyt_splade.SpladeFactory("/workspace/splade-cocondenser-ensembledistil")

def process_docs(dataset, splade, batch_size):
    ret = []

    indexer = splade.indexing()
    for chunk in tqdm(chunked(dataset.get_corpus_iter(), batch_size), 'Splade Indexing'):
        tmp = indexer(chunk)
        del tmp['text']
        ret += [tmp]

    return pd.concat(ret)

def splade_query_to_pyterrier_query(toks, mult=100):
    from pyt_splade import _matchop
    return ' '.join( _matchop(k, v * mult) for k, v in sorted(toks.items(), key=lambda x: (-x[1], x[0])))

def rescore_tokens(doc, mult=100):
    doc['toks'] = {k: floor(v*mult) for k,v in doc['toks'].items() if v > (1/mult)}
    return doc

def main(args):
    ensure_pyterrier_is_loaded()
    dataset = pt.get_dataset(f'irds:{args.input}')
    print('Step 1: Load splade.')
    splade = get_splade()
    
    print('Step 2: Process documents.')
    documents = process_docs(dataset, splade, args.batch)
    documents.to_json(f'{args.output}/documents.jsonl.gz', lines=True, orient='records')

    print('Step 4: Index')
    iter_indexer = pt.IterDictIndexer(f"{args.output}/spladeindex", meta={'docno': 150}, threads=1, pretokenised=True, verbose=True, overwrite=True)
    iter_indexer.index((rescore_tokens({'docno': i['docno'], 'toks': i['toks']}) for _, i in documents.iterrows()))
    
    print('Done.')

if __name__ == '__main__':
    args = parse_args()
    main(args)
