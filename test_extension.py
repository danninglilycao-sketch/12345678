"""Synthetic correctness checks; these are not real-data experiment results."""
import copy, json, tempfile
from pathlib import Path
import h5py
import numpy as np
import torch
from torch import nn
from baseline_core import ABMIL
from dataloader import mil_collate_fn
from utils import Struct
from research_extension import perturb_features, consistency_loss, train_one_epoch, run_dataset, make_comparison

def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    x=torch.randn(2,5,6); mask=torch.tensor([[1,1,1,1,1],[1,1,0,0,0]],dtype=torch.bool)
    noisy=perturb_features(x,mask,0.05)
    assert torch.equal(noisy[~mask],x[~mask])
    assert not torch.equal(noisy[mask],x[mask])
    assert torch.equal(perturb_features(x,mask,0),x)
    z=torch.randn(2,3,requires_grad=True); zn=torch.randn(2,3,requires_grad=True)
    assert abs(consistency_loss(z,z).item())<1e-7
    torch.testing.assert_close(consistency_loss(z,zn),consistency_loss(zn,z))
    loss=consistency_loss(z,zn);assert loss>=0
    loss.backward();assert z.grad.abs().sum()>0 and zn.grad.abs().sum()>0
    conf=Struct(D_feat=6,D_inner=8,D_attn=4,n_class=3,droprate=0.)
    model=ABMIL(conf)
    torch.testing.assert_close(model(x,mask=mask)[1], model(x[1:2,:2])[0])
    batch={'input':x,'label':torch.tensor([0,1]),'mask':mask}
    before=copy.deepcopy(model.state_dict())
    result=train_one_epoch(model,nn.CrossEntropyLoss(),[batch],torch.optim.AdamW(model.parameters(),lr=0.01),'cpu',0.05,0.1)
    assert all(np.isfinite(v) for v in result.values())
    assert any(not torch.equal(before[k],model.state_dict()[k]) for k in before)
    # Full training/selection/test/save pipeline with synthetic slides for all 3 dataset identifiers.
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);(root/'config').mkdir()
        for ds,name,classes in [('C16','camelyon16',2),('C17','camelyon17',4),('BRACS','bracs',7)]:
            (root/'feature'/ds).mkdir(parents=True); (root/'results'/ds).mkdir(parents=True)
            splits={k:[f'{k}_{c}' for c in range(classes)] for k in ['train','val','test']}
            (root/'results'/ds/'splits.json').write_text(json.dumps(splits))
            with h5py.File(root/'feature'/ds/'patch_feats_pretrain_medical_ssl.h5','w') as h:
                for names in splits.values():
                    for c,s in enumerate(names):
                        g=h.create_group(s);g.create_dataset('feat',data=np.random.default_rng(c).normal(size=(c+2,6)).astype('float32'));g.attrs['label']=c
            cfg={'dataset':ds,'D_feat':6,'D_inner':8,'D_attn':4,'n_class':classes,'droprate':0.,'seed':42,'B':1,'n_worker':0,'lr':0.001,'wd':0.00001,'train_epoch':1}
            import yaml
            (root/'config'/f'{name}_medical_ssl_config.yml').write_text(yaml.safe_dump(cfg))
            run=root/'results_extension'/'synthetic';run.mkdir(parents=True,exist_ok=True)
            summary=run_dataset(root,ds,run,[0.01],[0.1],epochs=1)
            assert summary['status']=='completed'
            assert (run/ds/'sigma_0.01_lambda_0.1'/'loss_curves.png').exists()
            assert summary['test_metrics'].keys() >= {'acc','auc','f1','loss'}
    print('PASS: masks, zero noise, symmetric KL, both-path gradients, padding-invariant predictions, optimizer update, synthetic C16/C17/BRACS pipelines. NOT real-data evaluation.')

if __name__=='__main__':main()
