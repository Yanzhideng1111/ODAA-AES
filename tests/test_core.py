import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from odaa_aes.config import load_config
from odaa_aes.data import (Essay, SegmentedEssayDataset, outer_split,
                           make_collate_fn, SourceScoreTransform, normalize, spec_for)
from odaa_aes.model import (ODAA_AES, conditional_weights, compute_losses,
                            grad_reverse, ordinal_loss, sigmoid_grl_schedule)
from odaa_aes.metrics import per_prompt_metrics, scores_from_normalized


class ToyTokenizer:
    pad_token_id = 0
    def __call__(self, text, add_special_tokens=False, truncation=False):
        return {'input_ids':[3+len(x)%20 for x in text.split()]}
    def num_special_tokens_to_add(self,pair=False):return 2
    def build_inputs_with_special_tokens(self,chunk):return [1]+chunk+[2]


class ToyEncoder(nn.Module):
    def __init__(self,h=16):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=h)
        self.embed = nn.Embedding(48,h)
    def forward(self,input_ids,attention_mask):
        return SimpleNamespace(last_hidden_state=self.embed(input_ids))


@pytest.fixture
def cfg():
    c=load_config('configs/asap.yaml')
    c['model']['max_tokens']=16
    c['model']['max_segments']=3
    c['model']['overlap']=4
    c['model']['hidden_size']=12
    c['training']['amp']=False
    return c


def test_rubric_and_target_holdout(cfg):
    essay=[Essay('e1','1','one two three',7), Essay('e2','2','four five',3)]
    a,b=outer_split(essay,'2')
    assert [x.essay_id for x in a]==['e1']
    assert [x.essay_id for x in b]==['e2']
    assert normalize(7, spec_for(cfg,'1'))==pytest.approx(.5)
    assert scores_from_normalized(np.array([0.5]),spec_for(cfg,'1'))[0]==7


def test_windows_and_padding(cfg):
    sentences=[Essay('e1','1',' '.join(['hi']*35),7),Essay('e2','2','abc',3)]
    ds=SegmentedEssayDataset(sentences,ToyTokenizer(),cfg)
    assert len(ds[0]['input_ids'])==3
    batch=make_collate_fn(ToyTokenizer())([ds[0],ds[1]])
    assert batch['input_ids'].shape[0:2]==(2,3)
    assert batch['segment_mask'][1].sum()==1


def test_soft_anchors_and_grl():
    anchors=torch.linspace(0,1,5)
    z=conditional_weights(torch.tensor([0.2,0.8]),anchors,.15)
    assert torch.allclose(z.sum(-1),torch.ones(2),atol=1e-5)
    assert z[0].argmax().item()==1
    assert z[1].argmax().item()==3
    assert conditional_weights(torch.tensor([0.2]),anchors,.15,'hard').sum()==1
    x=torch.tensor([2.0],requires_grad=True)
    (grad_reverse(x)*3).sum().backward()
    assert x.grad.item()==-3.
    assert sigmoid_grl_schedule(0)==pytest.approx(0.)
    assert .98 < sigmoid_grl_schedule(1) < 1.


def test_forward_and_backward(cfg):
    items=[Essay('e1','1','this is a sample essay',9),
           Essay('e2','2','yet another example here',2),
           Essay('e3','1','many more essay words here',4)]
    ds=SegmentedEssayDataset(items,ToyTokenizer(),cfg)
    batch=make_collate_fn(ToyTokenizer())([ds[i] for i in range(3)])
    model=ODAA_AES(cfg,2,encoder=ToyEncoder())
    result=model(batch['input_ids'],batch['attention_mask'],batch['segment_mask'])
    assert result['score'].shape == (3,)
    assert result['ocaa_logits'].shape==(3,5,2)
    loss,details=compute_losses(result,batch['score'],torch.tensor([0,1,0]),model,cfg,.5)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.proj.weight.grad is not None
    assert details['pairs'] == 3


def test_single_source_disables_adversaries(cfg):
    m=ODAA_AES(cfg,1,encoder=ToyEncoder())
    assert not m.use_gpaa and not m.use_ocaa
    assert m.use_ospl


def test_ordinal_loss_ranks_and_ties():
    y=torch.tensor([0.1,0.9,0.9])
    domain=torch.tensor([0,1,0])
    bad=torch.tensor([1.,0.,0.])
    good=torch.tensor([0.1,0.9,0.9])
    lbad,_=ordinal_loss(bad,y,domain)
    lgood,_=ordinal_loss(good,y,domain)
    assert lgood < lbad


def test_source_score_transform_no_target_fitting():
    x=SourceScoreTransform('source_percentile').fit(np.array([0.1,0.2,0.3,0.9]))
    z=x.transform(np.array([0.1,0.9]))
    assert np.array_equal(z,np.array([.25,1.]))
    assert len(x.x)==4


def test_qwk_perfect_prediction(cfg):
    spec=spec_for(cfg,'1')
    true=np.array([2,4,7,12],dtype=float)
    pred=(true-spec['min'])/(spec['max']-spec['min'])
    met=per_prompt_metrics(true,pred,spec)
    assert met['qwk']==pytest.approx(1.)
    assert met['mae']==pytest.approx(0.)
