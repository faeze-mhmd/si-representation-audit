                                                                                

import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
import warnings

for key in ('LOKY_MAX_CPU_COUNT','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
import scipy
import sklearn
from threadpoolctl import threadpool_limits
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from si_audit.baselines import classification_summary
from si_audit.cnn import build_fashion_cnn
from si_audit.fashion_data import load_training_pool,validate_manifest,digest,_load_pair
from si_audit.fashion_final import claim_once,checked_readout
from si_audit.fashion_training import cnn_from_bundle,prepare_data,evaluate_cnn,extract_cnn,save_json
from si_audit.fashion_readouts import fixed_readouts,predict_batched
from si_audit.final_evaluation import require,rank_agreement,mean_sample_sd
from si_audit.training import model_state_hash


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def prepare(data_dir,output):
                                                                                       
    folder=ROOT/'results/fashion/experiment'
    index=read(folder/'experiment_index.json')
    config,provenance=index['config'],index['provenance']
    plan_path=ROOT/'results/fashion/FINAL_EVALUATION_PLAN.md'
    plan=plan_path.read_text(encoding='utf-8')
    expected_runs={f'fashion_{s}_seed{seed}' for s in config['budgets'] for seed in config['seeds']}
    require(index['completed'] and not index['test_evaluated'] and len(index['runs'])==6 and
            {r['run_id'] for r in index['runs']}==expected_runs,'wrong/incomplete validation stage')
    freeze_path=ROOT/'results/fashion/TRAINING_FREEZE.json'
    require(sha(freeze_path)==provenance['training_freeze_sha256'],'training freeze changed')
    freeze=read(freeze_path)
    require(freeze['provenance']=={k:v for k,v in provenance.items() if k!='training_freeze_sha256'},'freeze provenance changed')
    require(sha(ROOT/'results/fashion/preflight/measurement_gate.json')==freeze['measurement_gate_sha256'],'cost gate changed')
    for relative,value in provenance['source_sha256'].items():
        require(sha(ROOT/relative)==value,'training source changed: '+relative)
    baseline_path=ROOT/'results/fashion/baselines/baseline_results.json'
    require(sha(baseline_path)==provenance['baseline_results_sha256'],'baseline results changed')
    baseline=read(baseline_path)
    for relative,value in baseline['provenance']['source_sha256'].items():
        require(sha(ROOT/relative)==value,'baseline source changed: '+relative)
    analysis=read(folder/'validation_analysis.json')
    require(not analysis['verification']['test_evaluated'] and analysis['verification']['models_reloaded']==6,'wrong analysis stage')
    require(sha(ROOT/'scripts/analyze_fashion.py')==analysis['source_sha256'],'validation analysis source changed')
    prediction_path=folder/'validation_predictions.csv'
    require(sha(prediction_path)==analysis['validation_predictions_sha256'],'validation predictions changed')
    with prediction_path.open(encoding='utf-8',newline='') as handle:
        reader=csv.DictReader(handle)
        columns=reader.fieldnames
        saved=list(reader)
    manifest_path=ROOT/'results/fashion/data/split_manifest.json'
    require(sha(manifest_path)==provenance['manifest_file_sha256'],'split manifest bytes changed')
    manifest=read(manifest_path)
    require(manifest['manifest_sha256']==provenance['manifest_sha256'] and
            digest({k:v for k,v in manifest.items() if k!='manifest_sha256'})==manifest['manifest_sha256'],'split manifest content changed')
    pool=load_training_pool(data_dir)
    validate_manifest(pool,manifest,config)
    data={s:prepare_data(pool,manifest,s) for s in config['budgets']}
    val_y=data['A']['validation']['y'].numpy()
    require(len(saved)==6000 and len(columns)==26,'wrong validation prediction shape')
    require([int(r['sample_id']) for r in saved]==manifest['validation_rows'],'validation IDs changed')
    require([int(r['truth']) for r in saved]==val_y.tolist(),'validation labels changed')
    val_predictions={name:np.array([int(r[name]) for r in saved]) for name in columns[2:]}
    models,records,ready={}, {}, []
    for entry in index['runs']:
        run=entry['run_id']
        record=read(folder/entry['record'])
        require(record['completed'] and not record['test_evaluated'] and record['last_epoch']==config['epochs']==30,'wrong neural stage')
        require(record['config']==config and record['provenance']==provenance,'CNN metadata changed')
        fingerprint=digest({'config':config,'provenance':provenance,'setting':record['setting'],'seed':record['seed']})
        require(record['fingerprint']==fingerprint,'CNN fingerprint changed')
        require([h['epoch'] for h in record['history']]==list(range(31)) and len(record['epoch_order_sha256'])==30,'epoch history changed')
        require(model_state_hash(build_fashion_cnn(record['seed']))==record['initial_model_state_sha256'],'initialization changed')
        bundle=read(folder/entry['bundle'])
        require(bundle['metadata']['fingerprint']==fingerprint and bundle['metadata']['epoch']==30,'bundle metadata changed')
        require(bundle['payload_sha256']==record['model_bundle_payload_sha256'],'bundle payload changed')
        model=cnn_from_bundle(bundle)
        require(model_state_hash(model)==record['final_model_state_sha256']==entry['final_model_state_sha256'] and
                record['final_model_state_sha256'] in plan,'unplanned neural weights')
        item=data[record['setting']]
        summary,predicted=evaluate_cnn(model,item['validation']['x'],item['validation']['y'],detailed=True)
        require(summary==record['final_prediction']['validation'] and np.array_equal(predicted,val_predictions[run]),'CNN validation drift')
        prior=next(r for r in analysis['table'] if r['run_id']==run)
        require(prior['validation']==summary,'analysis head summary drift')
        snapshot=record['snapshots']['30']
        scores={'train_si600':snapshot['train']['representations']['penultimate']['si'],
                'validation_si600':snapshot['validation']['representations']['penultimate']['si'],
                'validation_si2000':snapshot['validation']['sensitivity_penultimate']['si']}
        require(all(prior[k]==v for k,v in scores.items()),'saved SI predictor drift')
        parts={s:extract_cnn(model,v['x'],layers=('penultimate',))['penultimate'] for s,v in item.items()}
        previous={p['model']:p for p in record['probes']}
        for name,estimator in fixed_readouts(config,probes=True):
            key=run+'__'+name
            replay=checked_readout(estimator,parts['train'],item['train']['y'].numpy(),parts['validation'],val_y,previous[name],val_predictions[key])
            ready.append({'readout_id':key,'role':'probe','parent_run':run,'setting':record['setting'],'seed':record['seed'],
                          'estimator':estimator,'replay':replay})
        models[run],records[run]=model,record
        print('VALIDATION GATE '+run,flush=True)
    for seed in config['seeds']:
        require(len({r['initial_model_state_sha256'] for r in records.values() if r['seed']==seed})==1,'initialization not paired')
    for setting,item in data.items():
        train_x=item['train']['x'].numpy().reshape(len(item['train']['y']),784)
        val_x=item['validation']['x'].numpy().reshape(len(val_y),784)
        for name,estimator in fixed_readouts(config):
            key=f'fashion_{setting}__input_{name}'
            expected=next(r for r in baseline['models'] if r['readout_id']==key)
            replay=checked_readout(estimator,train_x,item['train']['y'].numpy(),val_x,val_y,expected,val_predictions[key])
            ready.append({'readout_id':key,'role':'input_baseline','parent_run':None,'setting':setting,'seed':None,
                          'estimator':estimator,'replay':replay})
        print('VALIDATION GATE input '+setting,flush=True)
    require(len(models)+len(ready)==24 and set(val_predictions)==set(models)|{r['readout_id'] for r in ready},'wrong readout set')
    audit=read(ROOT/'results/fashion/data/data_audit.json')
    test_provenance=audit['official_test']['provenance']
    require(all(v['sha256'] in plan and v['md5'] in plan for v in test_provenance['files'].values()),'unplanned test file')
    gate={'passed_at_utc':datetime.now(timezone.utc).isoformat(),'validation_readouts_reproduced':24,
          'exact_validation_predictions_reproduced':24,'neural_models_reloaded':6,'neural_refit':False,
          'test_performance_accessed':False,'test_provenance_expected':test_provenance,
          'frozen_training_provenance':provenance,
          'artifact_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in
              (plan_path,folder/'experiment_index.json',folder/'validation_analysis.json',prediction_path,baseline_path)},
          'evaluator_sha256':sha(__file__),'guard_module_sha256':sha(ROOT/'src/si_audit/fashion_final.py')}
    save_json(output/'validation_gate.json',gate)
    return models,records,ready,gate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    claim_once(args.output_dir,{'plan_sha256':sha(ROOT/'results/fashion/FINAL_EVALUATION_PLAN.md'),
               'evaluator_sha256':sha(__file__),'guard_module_sha256':sha(ROOT/'src/si_audit/fashion_final.py'),
               'authorization':'User explicitly requested execution of the frozen final evaluation in the current chat.'})
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    start=perf_counter()
    test_accessed=False
    try:
        with threadpool_limits(limits=1):
            models,records,ready,gate=prepare(args.data_dir,args.output_dir)
            gate_seconds=perf_counter()-start
            print('PASS all 24 exact validation readouts; opening official test.',flush=True)
            save_json(args.output_dir/'test_access_started.json',{'started_at_utc':datetime.now(timezone.utc).isoformat(),
                      'validation_gate_sha256':sha(args.output_dir/'validation_gate.json')})
            test_accessed=True
            test=_load_pair(args.data_dir,'t10k',10000,60000)
            require(test.provenance==gate['test_provenance_expected'],'test provenance changed')
            x=torch.from_numpy(test.images.astype(np.float32)[:,None,:,:]/255)
            y=torch.from_numpy(test.labels.copy())
            predictions,outcomes,parts={},[],{}

            def preserve(outcome,prediction):
                key=outcome['readout_id']
                predictions[key]=prediction
                outcomes.append(outcome)
                save_json(args.output_dir/'readouts'/(key+'.json'),{**outcome,'sample_ids':test.sample_ids.tolist(),
                          'truth':test.labels.tolist(),'predictions':prediction.tolist()})
                print('TEST '+key+' accuracy='+str(outcome['test']['accuracy']),flush=True)

            for run,model in models.items():
                tick=perf_counter()
                summary,predicted=evaluate_cnn(model,x,y,detailed=True)
                parts[run]=extract_cnn(model,x,layers=('penultimate',))['penultimate']
                record=records[run]
                require(model_state_hash(model)==record['final_model_state_sha256'],'weights changed during inference')
                snapshot=record['snapshots']['30']
                scores={'train_si600':snapshot['train']['representations']['penultimate']['si'],
                        'validation_si600':snapshot['validation']['representations']['penultimate']['si'],
                        'validation_accuracy':record['final_prediction']['validation']['accuracy'],
                        'negative_validation_cross_entropy':-record['final_prediction']['validation']['cross_entropy'],
                        'training_budget':record['training_n'],
                        'validation_si2000':snapshot['validation']['sensitivity_penultimate']['si']}
                preserve({'readout_id':run,'role':'primary_head','setting':record['setting'],'seed':record['seed'],
                          'test':summary,'fixed_predictor_scores':scores,'test_predict_and_feature_seconds':perf_counter()-tick,
                          'warnings':[]},predicted)
            for item in ready:
                tick=perf_counter()
                features=parts[item['parent_run']] if item['role']=='probe' else x.numpy().reshape(len(y),784)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    predicted=predict_batched(item['estimator'],features)
                metadata={k:v for k,v in item.items() if k not in ('estimator','replay')}
                preserve({**metadata,'test':classification_summary(test.labels,predicted),
                          'test_predict_seconds':perf_counter()-tick,'validation_replay':item['replay'],
                          'warnings':[ *item['replay']['warnings'],*[str(w.message) for w in caught]]},predicted)
        require(len(predictions)==24,'missing final readouts')
        csv_path=args.output_dir/'predictions.csv'
        with csv_path.open('x',encoding='utf-8',newline='') as handle:
            writer=csv.writer(handle,lineterminator='\n')
            writer.writerow(['sample_id','truth',*predictions])
            for i in range(len(y)):
                writer.writerow([int(test.sample_ids[i]),int(test.labels[i]),*[int(p[i]) for p in predictions.values()]])
        heads=[r for r in outcomes if r['role']=='primary_head']
        groups={s:{m:mean_sample_sd([r['test'][m] for r in heads if r['setting']==s]) for m in ('accuracy','macro_f1')} for s in ('A','B')}
        pairs=[]
        for seed in (11,22,33):
            a,b=[next(r for r in heads if r['setting']==s and r['seed']==seed) for s in ('A','B')]
            pairs.append({'seed':seed,'test_accuracy_B_minus_A':b['test']['accuracy']-a['test']['accuracy']})
        keys=list(heads[0]['fixed_predictor_scores'])
        def ranks(items):
            return {k:rank_agreement([r['fixed_predictor_scores'][k] for r in items],[r['test']['accuracy'] for r in items]) for k in keys}
        selected=np.flatnonzero(predictions['fashion_A_seed11']!=test.labels)[:12]
        save_json(args.output_dir/'error_gallery.json',{'run_id':'fashion_A_seed11','policy':'first 12 test errors in ascending original test row',
                  'examples':[{'sample_id':int(test.sample_ids[i]),'test_row':int(i),'truth':int(test.labels[i]),
                               'predicted':int(predictions['fashion_A_seed11'][i]),'pixels':test.images[i].tolist()} for i in selected]})
        result={'completed_at_utc':datetime.now(timezone.utc).isoformat(),'completed':True,'test_evaluated':True,
                'scope':'Frozen Fashion-MNIST final test; no neural refit or test SI; separate from Optdigits',
                'plan_sha256':sha(ROOT/'results/fashion/FINAL_EVALUATION_PLAN.md'),
                'validation_gate_sha256':sha(args.output_dir/'validation_gate.json'),
                'evaluator_sha256':sha(__file__),'guard_module_sha256':sha(ROOT/'src/si_audit/fashion_final.py'),
                'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
                               'sklearn':sklearn.__version__,'torch':torch.__version__,'device':'cpu','threads':1},
                'test_provenance':test.provenance,'test_n':len(y),'outcomes':outcomes,
                'primary_group_summary':groups,'paired_B_minus_A':pairs,'six_model_rank_agreement':ranks(heads),
                'within_budget_rank_agreement':{s:ranks([r for r in heads if r['setting']==s]) for s in ('A','B')},
                'predictions_sha256':sha(csv_path),'validation_gate_seconds':gate_seconds,
                'runtime_after_imports_seconds':perf_counter()-start,'neural_refit':False,'test_si_computed':False,
                'tensor_sha256':{k:r['final_model_state_sha256'] for k,r in records.items()}}
        save_json(args.output_dir/'final_results.json',result)
        print(json.dumps({'groups':groups,'rank_agreement':result['six_model_rank_agreement'],
                          'runtime_after_imports_seconds':result['runtime_after_imports_seconds']},indent=2),flush=True)
    except Exception as error:
        save_json(args.output_dir/'failure.json',{'failed_at_utc':datetime.now(timezone.utc).isoformat(),
                  'type':type(error).__name__,'message':str(error),'test_access_started':test_accessed,
                  'policy':'Preserve this attempt. Log any repair; do not silently rerun or tune.'})
        raise


if __name__=='__main__':
    main()
