import argparse
import pyterrier as pt
from math import floor
from tira.third_party_integrations import ensure_pyterrier_is_loaded, persist_and_normalize_run

def parse_args():
    parser = argparse.ArgumentParser(prog='Retrieve with SPLADE via Pyterrier.')

    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)

    return parser.parse_args()

def rescore_tokens(doc, mult=100):
    doc['toks'] = {k: floor(v*mult) for k,v in doc['toks'].items() if v > (1/mult)}
    return doc

def get_splade():
    ensure_pyterrier_is_loaded()
    import pyt_splade
    return pyt_splade.SpladeFactory("/workspace/splade-cocondenser-ensembledistil")

def process_docs(dataset, splade):
    ret = splade.indexing()(dataset.get_corpus_iter())
    del ret['text']
    return ret

def process_queries(dataset, splade):
    ret = [{'docno': i['qid'], 'text': i['query']} for _, i in dataset.get_topics('title').iterrows()]
    ret = splade.indexing()(ret)
    ret['qid'] = ret['docno']
    del ret['docno']
    del ret['text']
    return ret

def splade_query_to_pyterrier_query(toks, mult=100):
    from pyt_splade import _matchop
    return ' '.join( _matchop(k, v * mult) for k, v in sorted(toks.items(), key=lambda x: (-x[1], x[0])))

def main(args):
    ensure_pyterrier_is_loaded()
    dataset = pt.get_dataset(f'irds:{args.input}')
    print('Step 1: Load splade.')
    splade = get_splade()
    
    print('Step 2: Process documents.')
    documents = process_docs(dataset, splade)
    documents.to_json(f'{args.output}/documents.jsonl.gz', lines=True, orient='records')

    print('Step 3: Process queries.')
    queries = process_queries(dataset, splade)
    queries.to_json(f'{args.output}/queries.jsonl.gz', lines=True, orient='records')

    print('Step 4: Index')
    iter_indexer = pt.IterDictIndexer("./index", meta={'docno': 150}, threads=1, pretokenised=True, verbose=True, overwrite=True)
    iter_indexer.index((rescore_tokens({'docno': i['docno'], 'toks': i['toks']}) for _, i in documents.iterrows()))

    print('Step 5: Retrieve')
    queries['query'] = queries['toks'].apply(splade_query_to_pyterrier_query)
    splade_retr = splade.query() >> pt.BatchRetrieve('./index', wmodel='Tf')
    run = splade_retr(queries)

    print('Step 5: Persist Run.')
    persist_and_normalize_run(run, output_file=args.output, system_name='SPLADE++-CoCondenser-EnsembleDistil', depth=1000)

if __name__ == '__main__':
    args = parse_args()
    main(args)