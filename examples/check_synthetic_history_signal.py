"""Causal same-concept historical-mean controls, with fixed Laplace smoothing."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from experiment_long_mem_topk_akt_synthetic import metrics
from pykt.models.long_mem_topk_akt_workflow import write_json


def main(data,output):
    data=Path(data)
    manifest=json.loads((data/'manifest.json').read_text())
    meta=json.loads((data/'diagnostic_labels.json').read_text())
    predictions={'block_only':[],'all_history':[]}
    with (data/'test.csv').open() as f:
        for row in csv.DictReader(f):
            concepts=list(map(int,row['concepts'].split(',')))
            responses=list(map(int,row['responses'].split(',')))
            history={mode:defaultdict(lambda:[0,0]) for mode in predictions}
            for t,(c,r) in enumerate(zip(concepts,responses)):
                if t%manifest['block_len']==0: history['block_only'].clear()
                for mode in predictions:
                    correct,count=history[mode][c]
                    # Predict before observing the current response.
                    p=(correct+1)/(count+2)
                    if t%manifest['block_len']:
                        predictions[mode].append(dict(uid=row['uid'],position=t,target=r,probability=p))
                    history[mode][c]=[correct+r,count+1]
    result={}
    for mode,rows in predictions.items():
        result[mode]={
            'all':metrics(rows),
            'return_block':metrics([r for r in rows if r['position']>=manifest['return_start']]),
            'first_return':metrics([r for r in rows if meta[r['uid']]['first_return'][r['position']]]),
        }
    write_json(output,dict(results=result,formula='(past same-concept correct + 1)/(past same-concept count + 2)',
                          note='Fixed causal non-neural controls; no latent mastery or current/future response used.'))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.data,a.output)
