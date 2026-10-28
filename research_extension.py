"""Proposal 2: masked Gaussian feature perturbation + symmetric KL consistency."""
import argparse, copy, csv, json, math, os, platform, hashlib
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from baseline_core import ABMIL, evaluate
from dataloader import HDF5FeatureDataset, mil_collate_fn
from utils import load_config, seed_everything, infer_feature_dim

CONFIGS = {'C16':'camelyon16', 'C17':'camelyon17', 'BRACS':'bracs'}

def perturb_features(x, mask, sigma):
    if sigma < 0: raise ValueError('sigma must be nonnegative')
    return x + sigma * torch.randn_like(x) * mask.unsqueeze(-1).to(x.dtype)

def consistency_loss(logits, noisy_logits):
    # Both paths receive gradients; no teacher or stop-gradient.
    logp, logq = F.log_softmax(logits, -1), F.log_softmax(noisy_logits, -1)
    p, q = logp.exp(), logq.exp()
    return 0.5 * ((p * (logp-logq)).sum(-1) + (q * (logq-logp)).sum(-1)).mean()

def train_one_epoch(model, criterion, data_loader, optimizer, device, sigma, weight):
    model.train()
    sums = dict(loss=0., ce_loss=0., consistency_loss=0., acc=0.)
    count = 0
    for data in data_loader:
        x = data['input'].to(device, dtype=torch.float32)
        y = data['label'].to(device, dtype=torch.long)
        mask = data['mask'].to(device)
        optimizer.zero_grad(set_to_none=True)
        z = model(x, mask=mask)
        zn = model(perturb_features(x, mask, sigma), mask=mask)
        ce = 0.5 * (criterion(z, y) + criterion(zn, y))
        con = consistency_loss(z, zn)
        loss = ce + weight * con
        loss.backward()
        optimizer.step()
        n = len(y); count += n
        for k,v in [('loss',loss),('ce_loss',ce),('consistency_loss',con)]: sums[k] += v.item()*n
        sums['acc'] += (z.argmax(-1)==y).sum().item()*100
    return {k:v/count for k,v in sums.items()}

def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2))

def fit(conf, train, val, out, sigma, weight, device):
    out.mkdir(parents=True, exist_ok=False)
    seed_everything(int(conf.seed))
    model = ABMIL(conf).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=conf.lr, weight_decay=conf.wd)
    criterion = nn.CrossEntropyLoss()
    loader_args = dict(batch_size=int(conf.B), num_workers=int(conf.n_worker),
                       collate_fn=mil_collate_fn, pin_memory=device.type=='cuda')
    train_loader = DataLoader(train, shuffle=True, **loader_args)
    val_loader = DataLoader(val, shuffle=False, **loader_args)
    history=[]; best=-float('inf'); best_epoch=None
    for epoch in range(int(conf.train_epoch)):
        tr=train_one_epoch(model,criterion,train_loader,optimizer,device,sigma,weight)
        va=evaluate(model,criterion,val_loader,device,conf)
        score=va['auc']+va['f1']
        if not math.isfinite(score): raise ValueError('Nonfinite validation score')
        if score>best:
            best=score; best_epoch=epoch+1
            torch.save({'model_state_dict':model.state_dict(),'config':vars(conf),
                        'sigma':sigma,'lambda':weight,'best_epoch':best_epoch},out/'best_model.pt')
        row={'epoch':epoch+1,**{'train_'+k:v for k,v in tr.items()},**{'val_'+k:v for k,v in va.items()}}
        history.append(row)
        with (out/'history.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerows(history)
        print(f'{out.name} epoch {epoch+1}: train={tr["loss"]:.4f}, val={va["loss"]:.4f}, score={score:.4f}',flush=True)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7,4))
    for k in ['train_ce_loss','val_loss']: plt.plot([r['epoch'] for r in history],[r[k] for r in history],label=k)
    plt.xlabel('Epoch');plt.ylabel('Cross-entropy');plt.legend();plt.tight_layout()
    plt.savefig(out/'loss_curves.png',dpi=160);plt.close()
    write_json(out/'selection.json',{'validation_score':best,'best_epoch':best_epoch,'sigma':sigma,'lambda':weight})
    return best

def run_dataset(root, dataset, run_dir, sigmas, weights, epochs=None):
    conf=load_config(str(root/'config'/f'{CONFIGS[dataset]}_medical_ssl_config.yml'))
    conf.data_dir=str(root/'feature'/dataset/'patch_feats_pretrain_medical_ssl.h5')
    conf.dataset_csv_dir=str(root/'dataset_csv')
    if epochs: conf.train_epoch=epochs
    # Use the actual saved baseline splits, including their ordering.
    splits=json.loads((root/'results'/dataset/'splits.json').read_text())
    sets=[set(splits[k]) for k in ['train','val','test']]
    if any(sets[i]&sets[j] for i,j in [(0,1),(0,2),(1,2)]): raise ValueError('Overlapping splits')
    if any(len(splits[k])!=len(set(splits[k])) for k in splits): raise ValueError('Duplicate slides')
    train,val,test=[HDF5FeatureDataset(conf.data_dir,splits[k],conf) for k in ['train','val','test']]
    conf.D_feat=infer_feature_dim(train)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dest=run_dir/dataset;dest.mkdir(parents=True,exist_ok=True)
    signature={'config':vars(conf),'sigmas':sigmas,'weights':weights,
               'splits_sha256':hashlib.sha256(json.dumps(splits,sort_keys=True).encode()).hexdigest(),
               'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    manifest=dest/'experiment_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != signature:
        raise ValueError('Experiment settings changed; use a new --run name')
    write_json(manifest,signature)
    write_json(dest/'splits.json',splits)
    candidates=[]
    for sigma in sigmas:
        for weight in weights:
            candidate=dest/f'sigma_{sigma:g}_lambda_{weight:g}'
            if (candidate/'selection.json').exists():
                score=json.loads((candidate/'selection.json').read_text())['validation_score']
            else: score=fit(conf,train,val,candidate,sigma,weight,device)
            candidates.append((score,candidate))
    # No test metrics have been inspected during selection.
    score,selected=max(candidates,key=lambda t:t[0])
    ckpt=torch.load(selected/'best_model.pt',map_location=device,weights_only=False)
    model=ABMIL(conf).to(device);model.load_state_dict(ckpt['model_state_dict'])
    test_loader=DataLoader(test,batch_size=int(conf.B),shuffle=False,num_workers=int(conf.n_worker),collate_fn=mil_collate_fn)
    metrics=evaluate(model,nn.CrossEntropyLoss(),test_loader,device,conf)
    summary={'dataset':dataset,'selected_candidate':selected.name,'validation_score':score,
             'best_epoch':ckpt['best_epoch'],'sigma':ckpt['sigma'],'lambda':ckpt['lambda'],
             'test_metrics':metrics,'config':vars(conf),'torch_version':torch.__version__,
             'python_version':platform.python_version(),'device':str(device),
             'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None,
             'status':'completed','single_seed_limitation':True}
    write_json(dest/'metrics.json',summary)
    return summary

def make_comparison(root,run_dir):
    rows=[]
    for ds in CONFIGS:
        base=json.loads((root/'results'/ds/'metrics.json').read_text())['test_metrics']
        path=run_dir/ds/'metrics.json'
        new=json.loads(path.read_text())['test_metrics'] if path.exists() else None
        for metric in ['acc','auc','f1','loss']:
            rows.append({'dataset':ds,'metric':metric,'baseline':base[metric],
                         'extension':new[metric] if new else '',
                         'delta':new[metric]-base[metric] if new else '',
                         'status':'completed' if new else 'not_run'})
    with (run_dir/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# Experiment comparison','', 'Accuracy is in percent; accuracy deltas are percentage points. Other deltas use raw metric units.',
           'Single-seed comparisons do not establish statistical significance.','',
           '| Dataset | Metric | Baseline | Extension | Delta |','|---|---|---:|---:|---:|']
    for r in rows:
        e=f'{r["extension"]:.4f}' if r['extension']!='' else 'NOT RUN'
        d=f'{r["delta"]:+.4f}' if r['delta']!='' else '—'
        lines.append(f'| {r["dataset"]} | {r["metric"]} | {r["baseline"]:.4f} | {e} | {d} |')
    lines+=['','Interpret improvements/decreases after execution. Potential mechanisms (hypotheses, not established causes): regularization may reduce sensitivity; excessive noise may remove sparse diagnostic signals; class imbalance and dataset differences may affect macro-F1. No causal or generalization claim follows from an unrun experiment.']
    (run_dir/'comparison_report.md').write_text('\n'.join(lines))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',default=str(Path(__file__).parent))
    ap.add_argument('--datasets',nargs='+',choices=list(CONFIGS),default=list(CONFIGS))
    ap.add_argument('--run',default='consistency_v1');ap.add_argument('--sigmas',nargs='+',type=float,default=[0.01,0.05])
    ap.add_argument('--weights',nargs='+',type=float,default=[0.1,1.0]);ap.add_argument('--epochs',type=int)
    ap.add_argument('--report-only',action='store_true');args=ap.parse_args()
    if Path(args.run).name!=args.run or args.run in ['.','..']: ap.error('run must be a folder name')
    if any(v<=0 for v in args.sigmas) or any(v<0 for v in args.weights): ap.error('Require sigma>0 and lambda>=0')
    root=Path(args.project).resolve();os.chdir(root)
    run_dir=root/'results_extension'/args.run;run_dir.mkdir(parents=True,exist_ok=True)
    if not args.report_only:
        for ds in args.datasets:
            if (run_dir/ds/'metrics.json').exists():
                print(f'{ds}: completed; skipping. Use a new --run name for another experiment.');continue
            run_dataset(root,ds,run_dir,args.sigmas,args.weights,args.epochs)
            make_comparison(root,run_dir)
    make_comparison(root,run_dir)

if __name__=='__main__': main()
