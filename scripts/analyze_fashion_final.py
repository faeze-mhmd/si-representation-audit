                                                                                        

import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['LOKY_MAX_CPU_COUNT']='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from si_audit.baselines import classification_summary
from si_audit.fashion_data import _load_pair
from si_audit.fashion_training import save_json
from si_audit.final_evaluation import require,rank_agreement,mean_sample_sd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify(folder,data_dir):
    result=read(folder/'final_results.json')
    require(result['completed'] and result['test_evaluated'] and not result['neural_refit'] and not result['test_si_computed'],'wrong final stage')
    plan=ROOT/'results/fashion/FINAL_EVALUATION_PLAN.md'
    require(sha(plan)==result['plan_sha256'],'frozen plan changed')
    marker=read(folder/'evaluation_started.json')
    gate=read(folder/'validation_gate.json')
    access=read(folder/'test_access_started.json')
    require(marker['started_at_utc']<=gate['passed_at_utc']<=access['started_at_utc']<=result['completed_at_utc'],'event order changed')
    require(sha(folder/'validation_gate.json')==result['validation_gate_sha256']==access['validation_gate_sha256'],'validation gate changed')
    require(gate['validation_readouts_reproduced']==gate['exact_validation_predictions_reproduced']==24 and not gate['test_performance_accessed'],'gate incomplete')
    for name,key in [('scripts/evaluate_fashion_final.py','evaluator_sha256'),('src/si_audit/fashion_final.py','guard_module_sha256')]:
        require(sha(ROOT/name)==result[key]==gate[key]==marker[key],'execution source changed')
    for name,value in gate['artifact_sha256'].items():
        require(sha(ROOT/name)==value,'validation artifact changed: '+name)
    for name,value in gate['frozen_training_provenance']['source_sha256'].items():
        require(sha(ROOT/name)==value,'frozen training source changed: '+name)
    require(sha(ROOT/'results/fashion/TRAINING_FREEZE.json')==gate['frozen_training_provenance']['training_freeze_sha256'],'training freeze changed')
    require(sha(folder/'predictions.csv')==result['predictions_sha256'],'final prediction file changed')
    with (folder/'predictions.csv').open(encoding='utf-8',newline='') as handle:
        reader=csv.DictReader(handle)
        columns=reader.fieldnames
        rows=list(reader)
    require(len(rows)==result['test_n']==10000 and len(columns)==26,'wrong final CSV shape')
    test=_load_pair(data_dir,'t10k',10000,60000)
    require(test.provenance==result['test_provenance']==gate['test_provenance_expected'],'test data changed')
    require([int(r['sample_id']) for r in rows]==test.sample_ids.tolist() and [int(r['truth']) for r in rows]==test.labels.tolist(),'test IDs or truth changed')
    outcomes=result['outcomes']
    require(len(outcomes)==24 and set(columns[2:])=={r['readout_id'] for r in outcomes},'wrong final readouts')
    for item in outcomes:
        key=item['readout_id']
        predicted=np.array([int(r[key]) for r in rows])
        summary=classification_summary(test.labels,predicted)
        require(all(item['test'][k]==v for k,v in summary.items()),'final metrics differ: '+key)
        partial_path=folder/'readouts'/(key+'.json')
        if partial_path.exists():
            persisted=read(partial_path)
        else:
                                                                               
                                                                                
            partial_manifest=read(folder/'readout_manifest.json')
            persisted={**item,'sample_ids':test.sample_ids.tolist(),'truth':test.labels.tolist(),'predictions':predicted.tolist()}
            reconstructed=(json.dumps(persisted,indent=2)+'\n').encode('utf-8')
            require(hashlib.sha256(reconstructed).hexdigest()==partial_manifest['readouts'][key]['sha256'],'portable partial reconstruction changed')
        require(all(persisted[k]==v for k,v in item.items()),'partial/final outcome mismatch')
        require(persisted['predictions']==predicted.tolist() and persisted['truth']==test.labels.tolist() and persisted['sample_ids']==test.sample_ids.tolist(),'partial/final prediction mismatch')
        if item['role']=='primary_head':
            record=read(ROOT/'results/fashion/experiment'/key/'metrics.json')
            require(record['final_model_state_sha256']==result['tensor_sha256'][key],'tensor attribution changed')
            snapshot=record['snapshots']['30']
            scores={'train_si600':snapshot['train']['representations']['penultimate']['si'],
                    'validation_si600':snapshot['validation']['representations']['penultimate']['si'],
                    'validation_si2000':snapshot['validation']['sensitivity_penultimate']['si'],
                    'validation_accuracy':record['final_prediction']['validation']['accuracy'],
                    'negative_validation_cross_entropy':-record['final_prediction']['validation']['cross_entropy'],
                    'training_budget':record['training_n']}
            require(item['fixed_predictor_scores']==scores,'predictor score changed')
        else:
            if item['role']=='probe':
                original=next(p for p in read(ROOT/'results/fashion/experiment'/item['parent_run']/'metrics.json')['probes'] if p['model']==key.split('__')[-1])
            else:
                original=next(p for p in read(ROOT/'results/fashion/baselines/baseline_results.json')['models'] if p['readout_id']==key)
            require(all(item['validation_replay'][k]==original[k] for k in ('validation','parameters','iterations','warnings')),'readout replay changed')
            require(item['warnings']==original['warnings'],'unexpected prediction warning')
    heads=[r for r in outcomes if r['role']=='primary_head']
    groups={s:{m:mean_sample_sd([r['test'][m] for r in heads if r['setting']==s]) for m in ('accuracy','macro_f1')} for s in ('A','B')}
    require(groups==result['primary_group_summary'],'group summaries changed')
    def ranks(items):
        return {k:rank_agreement([r['fixed_predictor_scores'][k] for r in items],[r['test']['accuracy'] for r in items]) for k in heads[0]['fixed_predictor_scores']}
    require(ranks(heads)==result['six_model_rank_agreement'],'six-model ranks changed')
    require({s:ranks([r for r in heads if r['setting']==s]) for s in ('A','B')}==result['within_budget_rank_agreement'],'within-budget ranks changed')
    pairs=[{'seed':seed,'test_accuracy_B_minus_A':next(r['test']['accuracy'] for r in heads if r['seed']==seed and r['setting']=='B')-next(r['test']['accuracy'] for r in heads if r['seed']==seed and r['setting']=='A')} for seed in (11,22,33)]
    require(pairs==result['paired_B_minus_A'],'paired differences changed')
    gallery=read(folder/'error_gallery.json')
    key=gallery['run_id']
    require(key=='fashion_A_seed11','gallery model changed')
    errors=[i for i,r in enumerate(rows) if int(r[key])!=int(r['truth'])][:12]
    expected=[{'sample_id':int(test.sample_ids[i]),'test_row':i,'truth':int(test.labels[i]),'predicted':int(rows[i][key]),'pixels':test.images[i].tolist()} for i in errors]
    require(gallery['examples']==expected,'error gallery changed')
    return result,gallery


def report(result,output):
    heads=[r for r in result['outcomes'] if r['role']=='primary_head']
    lines=['# Fashion-MNIST — frozen final evaluation','',
        'Completed '+result['completed_at_utc']+'. All 24 predeclared readouts were evaluated on the official 10,000-image test set after exact reproduction of their saved validation predictions. This test is now inspected. No neural refit, checkpoint selection, test SI or post-test tuning was performed.','',
        '## Fixed design','',
        'One 9,802-parameter CNN, three paired initialization seeds (11/22/33), 30 epochs, A=3,000 and B=12,000 nested training images. Validation is 6,000 held-out original training images. Fixed SI600 train and validation scores are primary predictors; validation SI2000 is a declared sensitivity view. B has 2,820 updates versus A\'s 720, so its intervention combines data quantity and training exposure. The comparison does not isolate a matched-compute data effect.','',
        'The [pre-test plan](../FINAL_EVALUATION_PLAN.md), [training freeze](../TRAINING_FREEZE.json), [validation report](../experiment/VALIDATION_REPORT.md), [validation gate](validation_gate.json), [results](final_results.json), [all predictions](predictions.csv) and [saved-result verification](VERIFICATION.json) preserve the evidence. Training/validation artifacts retain their historical pre-test status.','',
        '## All six classifier heads','',
        '| Budget | Seed | Validation accuracy | Test accuracy | Test macro-F1 | Errors | Train SI600 | Validation SI600 | Validation SI2000 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in heads:
        t,s=r['test'],r['fixed_predictor_scores']
        errors=t['n']-sum(t['confusion_matrix'][i][i] for i in range(10))
        lines.append(f"| {r['setting']} | {r['seed']} | {s['validation_accuracy']:.2%} | {t['accuracy']:.2%} | {t['macro_f1']:.4f} | {errors} | {s['train_si600']:.2%} | {s['validation_si600']:.2%} | {s['validation_si2000']:.2%} |")
    lines+=['','| Budget | Mean test accuracy | Sample SD (percentage points) | Mean test macro-F1 |','|---|---:|---:|---:|']
    for s,g in result['primary_group_summary'].items():
        lines.append(f"| {s} | {g['accuracy']['mean']:.2%} | {100*g['accuracy']['sample_sd']:.3f} | {g['macro_f1']['mean']:.4f} |")
    lines+=['','Seed SD is descriptive, not a confidence interval. Paired B minus A differences (percentage points): '+', '.join(f"seed {r['seed']}: {100*r['test_accuracy_B_minus_A']:+.3f}" for r in result['paired_B_minus_A'])+'.','',
            '## All six input baselines and twelve frozen-feature probes','',
            '| Readout | Test accuracy | Macro-F1 | Fit/prediction warnings |','|---|---:|---:|---:|']
    for r in result['outcomes']:
        if r['role']!='primary_head':
            lines.append(f"| {r['readout_id']} | {r['test']['accuracy']:.2%} | {r['test']['macro_f1']:.4f} | {len(r['warnings'])} |")
    lines+=['','All six linear probes reached max_iter=1000 during training-only refitting, matching validation warnings. They are optimization-budget-limited probes; they are not fully optimized estimates of linear separability. No iterations, normalization or C values were adjusted. Complete warning text, parameters, iterations and timings are in the results and readout files. Input logistic regression had no convergence warning.','',
            '## Declared rank agreement with head test accuracy','',
            '| Saved predictor | All six models rho | Within A rho | Within B rho |','|---|---:|---:|---:|']
    def fmt(value):
        return 'undefined (constant)' if value is None else f'{value:.3f}'
    for key,value in result['six_model_rank_agreement'].items():
        lines.append('| '+key+' | '+fmt(value['spearman_rho'])+' | '+fmt(result['within_budget_rank_agreement']['A'][key]['spearman_rho'])+' | '+fmt(result['within_budget_rank_agreement']['B'][key]['spearman_rho'])+' |')
    lines+=['','Rho uses average ranks for ties. Pair counts are retained in JSON. There are only six related models, three seeds within each budget, one dataset, one architecture and one split. Between-budget separation can dominate six-model agreement. No p-values, significance, universal predictor ranking or independent-model claim is justified. SI is supervised; validation SI also uses validation labels. Larger SI subsets change neighbourhood geometry and do not create independent experiments. Optdigits and Fashion-MNIST are reported separately.','',
            'In this test, all three A heads fall below the A input logistic baseline (81.14%); two of three B heads exceed the B input logistic baseline (82.11%). Higher aggregate SI agreement does not imply a universal ordering: within A, validation SI600 and SI2000 decrease as head test accuracy increases (rho=-1 across only three seeds). Train SI600 and ordinary validation accuracy both have six-model rho=0.943. These observations do not establish an advantage for SI as a transferable performance predictor.','',
            '## Per-class recall of each head','',
            '| Run | T-shirt/top | Trouser | Pullover | Dress | Coat | Sandal | Shirt | Sneaker | Bag | Ankle boot |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in heads:
        lines.append('| '+r['readout_id']+' | '+' | '.join(f"{r['test']['per_class'][str(i)]['recall']:.2%}" for i in range(10))+' |')
    lines+=['','Complete confusion matrices and per-class precision/F1/support for all 24 readouts are retained in JSON.','',
            '![Accuracy and fixed SI](figures/si_vs_test.svg)','',
            '![Head confusion matrices](figures/confusion_matrices.svg)','',
            '![Fixed error gallery](figures/test_errors.svg)','',
            'The error gallery is the first twelve A/seed-11 errors in ascending test row order, as declared before evaluation. It is diagnostic and is not a random sample or a basis for model selection.','',
            '## Execution and limitations','',
            f"CPU, one thread. Validation reproduction gate: {result['validation_gate_seconds']:.3f} s; total evaluation after imports: {result['runtime_after_imports_seconds']:.3f} s. No neural training is included. Runtime includes training-only classifier refits, feature extraction, test prediction and artifact export. The existing environment passed 72 unit tests, including four new final-evaluation guards. No fresh-environment installation was tested.",'',
            'Saved predictions reproduce classification metrics for all 24 readouts, all group/paired summaries and all 18 rank comparisons without fitting or neural inference. Cross-entropy cannot be reconstructed from class predictions alone; it was computed from frozen-model logits by the evaluator. Verification is a software/artifact consistency check, not an independent replication. The portable [readout manifest](readout_manifest.json) permits exact reconstruction of the per-readout execution JSON from the full results and shared CSV; duplicate IDs/truth arrays are factored out. Original per-readout execution files remain in the local C: cache.','',
            'Official file hashes and format were checked. Exact duplicates/conflicts were absent in the prior 70,000-image audit; near duplicates and product/subject independence remain unverified. Benchmark limitations and limited seed count restrict transfer claims. Any later model or method change informed by these results is exploratory and needs new confirmation evidence.','',
            'No Git initialization, staging, commit, push or publication occurred in this milestone.','']
    (output/'FINAL_REPORT.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')


def figures(result,gallery,output):
    os.environ['MPLCONFIGDIR']=str(output/'mpl-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','svg.hashsalt':'fashion-final-v1','axes.spines.top':False,'axes.spines.right':False})
    dest=output/'figures'
    dest.mkdir(exist_ok=True)
    def save(fig,name):
        fig.savefig(dest/(name+'.svg'),metadata={'Date':None},facecolor='white')
        fig.savefig(dest/(name+'.png'),dpi=150,facecolor='white')
        plt.close(fig)
    heads=[r for r in result['outcomes'] if r['role']=='primary_head']
    fig,axes=plt.subplots(1,3,figsize=(11,4),layout='constrained')
    for ax,key,label in zip(axes,('train_si600','validation_si600','validation_si2000'),('Train SI (600)','Validation SI (600, primary)','Validation SI (2,000, sensitivity)')):
        for r in heads:
            s=r['setting']
            xy=(r['fixed_predictor_scores'][key],r['test']['accuracy'])
            ax.scatter(*xy,color='#1864ab' if s=='A' else '#d9480f',marker='o' if s=='A' else '^')
            offsets={'A11':(-28,-17),'A22':(7,-15),'A33':(-28,8),'B11':(7,6),'B22':(-28,8),'B33':(7,-16)}
            ax.annotate(s+str(r['seed']),xy,xytext=offsets[s+str(r['seed'])],textcoords='offset points',fontsize=8)
        ax.set(xlabel=label,ylabel='Official test head accuracy')
        ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=1))
        ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=1))
        ax.margins(x=.3,y=.25)
        ax.grid(alpha=.15)
    fig.suptitle('Six frozen CNNs: fixed pre-test SI versus official test accuracy')
    fig.supxlabel('Descriptive only: three related seeds per budget; more data also means more optimizer updates.',fontsize=9)
    save(fig,'si_vs_test')
    fig,axes=plt.subplots(2,3,figsize=(11,7),layout='constrained')
    for ax,r in zip(axes.flat,sorted(heads,key=lambda r:(r['setting'],r['seed']))):
        matrix=np.array(r['test']['confusion_matrix'])
        image=ax.imshow(matrix/matrix.sum(1,keepdims=True),vmin=0,vmax=1,cmap='Blues')
        ax.set(title=r['readout_id'],xlabel='Predicted class ID',ylabel='True class ID',xticks=range(10),yticks=range(10))
    fig.colorbar(image,ax=axes.ravel().tolist(),label='Recall fraction within true class',shrink=.8)
    fig.suptitle('Official test: row-normalized confusion; same scale for every head')
    save(fig,'confusion_matrices')
    fig,axes=plt.subplots(3,4,figsize=(8,6),layout='constrained')
    for ax,e in zip(axes.flat,gallery['examples']):
        ax.imshow(e['pixels'],cmap='gray_r',vmin=0,vmax=255,interpolation='nearest')
        ax.set_title(f"Row {e['test_row']}: {e['truth']} -> {e['predicted']}",fontsize=9)
        ax.axis('off')
    fig.suptitle('A, seed 11: first 12 official test errors\nFixed diagnostic selection; class IDs 0-9',fontsize=11)
    save(fig,'test_errors')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    result,gallery=verify(args.results_dir,args.data_dir)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    verification={'status':'PASS','verified_at_utc':datetime.now(timezone.utc).isoformat(),'readouts_verified':24,
                  'test_rows_verified':10000,'rank_comparisons_verified':18,'gallery_examples_verified':12,
                  'new_fit_or_neural_inference':False,'cross_entropy_reconstructed_from_classes':False,
                  'analysis_source_sha256':sha(__file__),'results_sha256':sha(args.results_dir/'final_results.json'),
                  'predictions_sha256':sha(args.results_dir/'predictions.csv')}
    save_json(args.output_dir/'VERIFICATION.json',verification)
    figures(result,gallery,args.output_dir)
    report(result,args.output_dir)
    print(json.dumps(verification,indent=2))


if __name__=='__main__':
    main()
