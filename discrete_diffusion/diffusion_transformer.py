#update matrix for larger dataset
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast
from tqdm import tqdm
from torch.nn import TransformerEncoderLayer, TransformerEncoder 
from discrete_diffusion.conditional_attention import Transformer
from datamodule import Batch
from einops import rearrange

eps = 1e-8

def sum_except_batch(x, num_dims=1):
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)

def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)

def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.to(t.device).gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)

def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    permute_order = (0, -1) + tuple(range(1, len(x.size())))
    x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x

def log_onehot_to_index(log_x):
    return log_x.argmax(1)



def alpha_schedule(time_step, att_1 = 0.99999, att_T = 0.000009, ctt_1 = 0.000009, ctt_T = 0.99999, type_classes=9, poi_classes=381):

    sep=5
    sep_1=sep-1
    att= np.concatenate((np.arange(0, time_step*sep_1//sep)/(time_step*sep_1//sep-1)*(0.0001 - 0.99999) + 0.99999,
    np.arange(0, time_step-time_step*sep_1//sep)/(time_step-time_step*sep_1//sep-1)*(0.000009- 0.00009) + 0.00009))
    att = np.concatenate(([1], att))
    at = att[1:]/att[:-1]

    att1= np.concatenate((np.arange(0, time_step*sep_1//sep)/(time_step*sep_1//sep-1)*(0.9999 - 0.99999) + 0.99999,
    np.arange(0, time_step-time_step*sep_1//sep)/(time_step-time_step*sep_1//sep-1)*(0.000009- 0.9999) + 0.9999))
    att1 = np.concatenate(([1], att1))
    at1 = att1[1:]/att1[:-1]

    ctt= np.concatenate((np.arange(0, time_step*sep_1//sep)/(time_step*sep_1//sep-1)*(0.00009 - 0.000009) + 0.000009,
    np.arange(0, time_step-time_step*sep_1//sep)/(time_step-time_step*sep_1//sep-1)*(0.9999- 0.0001) + 0.0001))
    ctt = np.concatenate(([0], ctt))
    one_minus_ctt = 1 - ctt
    one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
    ct = 1-one_minus_ct

    ctt1= np.concatenate((np.arange(0, time_step*sep_1//sep)/(time_step*sep_1//sep-1)*(0.00009 - 0.000009) + 0.000009,
    np.arange(0, time_step-time_step*sep_1//sep)/(time_step-time_step*sep_1//sep-1)*(0.9998- 0.00009) + 0.00009))
    ctt1 = np.concatenate(([0], ctt1)) 
    one_minus_ctt1 = 1 - ctt1 
    one_minus_ct1 = one_minus_ctt1[1:] / one_minus_ctt1[:-1]
    ct1 = 1-one_minus_ct1 

    att = np.concatenate((att[1:], [1]))
    ctt = np.concatenate((ctt[1:], [0]))
    att1 = np.concatenate((att1[1:], [1]))
    ctt1 = np.concatenate((ctt1[1:], [0]))
    btt1 = (1-att1-ctt1) / type_classes
    btt2 = (1-att-ctt)

    bt1 = (1-at1-ct1) / type_classes 
    btt2 = np.concatenate(([0], btt2))
    one_minus_btt2 = 1 - btt2
    one_minus_bt = one_minus_btt2[1:] / one_minus_btt2[:-1]
    bt = 1-one_minus_bt
    btt2 = (1-att-ctt)/poi_classes

    bt=np.concatenate((bt[:time_step*sep_1//sep],at1[time_step*sep_1//sep:]/poi_classes))
    at=np.concatenate((at[:time_step*sep_1//sep],(1-ct-bt*poi_classes)[time_step*sep_1//sep:])).clip(min=1e-30)
    ct=np.concatenate(((1-at-bt)[:time_step*sep_1//sep],ct[time_step*sep_1//sep:])).clip(min=1e-30)

    return at,at1, bt,bt1, ct,ct1, att,att1, btt1,btt2, ctt,ctt1 


class ConditionEmbeddingModel(nn.Module):
    def __init__(
        self,
        cond_token_num = 200, # total number of all of condition tokens
        emb_dims = 256,
        num_condition_types = 6,
        max_position_embeddings = 3000
    ):
        super().__init__()
        self.token_num = cond_token_num
        self.emb_dims = emb_dims
        self.num_condition_types = num_condition_types + 1 # hour in day are discretized as additional condition
        self.max_position_embeddings = max_position_embeddings
        self.encoder = nn.Embedding(self.token_num, self.emb_dims)
        self.input_up_proj =  nn.Sequential(
            nn.Linear(self.num_condition_types * self.emb_dims, self.emb_dims),
            nn.ReLU(),
            nn.Linear(self.emb_dims, self.emb_dims)
        )
        self.position_embeddings = nn.Embedding(self.max_position_embeddings, self.emb_dims)
        self.register_buffer("position_ids", torch.arange(self.max_position_embeddings).expand((1, -1)))
        encoder_layer = TransformerEncoderLayer(d_model=self.emb_dims, nhead=4, batch_first=True)
        self.condition_transformers = TransformerEncoder(encoder_layer, num_layers=3)


    def forward(self, batch):
        seq_length = batch.time.size(1)
        position_ids = self.position_ids[:, : seq_length ]
        time_embeddings = self.encoder(batch.time.long()+1)
        condition1_embeddings = self.encoder(batch.condition1)
        condition2_embeddings = self.encoder(batch.condition2)
        condition3_embeddings = self.encoder(batch.condition3)
        condition4_embeddings = self.encoder(batch.condition4)
        condition5_embeddings = self.encoder(batch.condition5)
        condition6_embeddings = self.encoder(batch.condition6)

        condition_embeddings = self.input_up_proj(torch.cat([time_embeddings,condition1_embeddings,condition2_embeddings,\
            condition3_embeddings,condition4_embeddings,condition5_embeddings,condition6_embeddings],dim=-1))
        condition_embeddings = self.position_embeddings(position_ids) + condition_embeddings
        encoded_conditions = self.condition_transformers(condition_embeddings)
        return encoded_conditions



class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        *,
        diffusion_step=200,
        alpha_init_type='alpha1',
        num_condition_types=6,
        type_classes=9,
        poi_classes=3477,
        num_spectial=4,
        num_classes=None,

    ):
        super().__init__()  

        self.schedule_type=alpha_init_type
        self.amp = False
        self.num_condition_types = num_condition_types

        self.num_classes = type_classes+poi_classes+num_spectial+2 
        self.type_classes = type_classes 
        self.num_spectial = num_spectial 
        self.poi_classes = poi_classes 
        self.transformer = Transformer(tgt_vocab_size=self.num_classes,num_spectial=self.num_spectial,type_classes=self.type_classes,poi_classes=self.poi_classes)
        self.loss_type = 'vb_stochastic'
        self.num_timesteps = diffusion_step
        self.parametrization = 'x0'

        at,at1, bt,bt1, ct,ct1, att,att1, btt1,btt2, ctt,ctt1 = alpha_schedule(self.num_timesteps, type_classes=self.type_classes, poi_classes = self.poi_classes)

 
        at1 = torch.tensor(at1.astype('float64'))
        bt1 = torch.tensor(bt1.astype('float64'))
        ct1 = torch.tensor(ct1.astype('float64'))
        log_at1 = torch.log(at1).clamp(-70,0)
        log_bt1 = torch.log(bt1).clamp(-70,0)
        log_ct1 = torch.log(ct1).clamp(-70,0)

        att1 = torch.tensor(att1.astype('float64'))
        btt1 = torch.tensor(btt1.astype('float64'))
        ctt1 = torch.tensor(ctt1.astype('float64'))
        log_cumprod_at1 = torch.log(att1).clamp(-70,0)
        log_cumprod_bt1 = torch.log(btt1).clamp(-70,0)
        log_cumprod_ct1 = torch.log(ctt1).clamp(-70,0) 

        log_1_min_ct1 = log_1_min_a(log_ct1) # torch.log(1 - a.exp() + 1e-40)
        log_1_min_cumprod_ct1 = log_1_min_a(log_cumprod_ct1)
        assert log_add_exp(log_ct1, log_1_min_ct1).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_ct1, log_1_min_cumprod_ct1).abs().sum().item() < 1.e-5
        self.register_buffer('log_ct1', log_ct1.float())
        self.register_buffer('log_bt1', log_bt1.float())
        self.register_buffer('log_at1', log_at1.float())
        self.register_buffer('log_cumprod_at1', log_cumprod_at1.float())
        self.register_buffer('log_cumprod_bt1', log_cumprod_bt1.float())
        self.register_buffer('log_cumprod_ct1', log_cumprod_ct1.float())
        self.register_buffer('log_1_min_ct1', log_1_min_ct1.float())
        self.register_buffer('log_1_min_cumprod_ct1', log_1_min_cumprod_ct1.float())

        at = torch.tensor(at.astype('float64'))
        bt = torch.tensor(bt.astype('float64'))
        ct = torch.tensor(ct.astype('float64'))
        log_at = torch.log(at).clamp(-70,0)
        log_bt = torch.log(bt).clamp(-70,0)
        log_ct = torch.log(ct).clamp(-70,0)
        att = torch.tensor(att.astype('float64'))
        btt2 = torch.tensor(btt2.astype('float64'))
        ctt = torch.tensor(ctt.astype('float64'))
        log_cumprod_at = torch.log(att).clamp(-70,0)
        log_cumprod_bt = torch.log(btt2).clamp(-70,0)
        log_cumprod_ct = torch.log(ctt).clamp(-70,0)

        log_1_min_ct = log_1_min_a(log_ct)
        log_1_min_cumprod_ct = log_1_min_a(log_cumprod_ct)

        assert log_add_exp(log_ct, log_1_min_ct).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_ct, log_1_min_cumprod_ct).abs().sum().item() < 1.e-5
        # Convert to float32 and register buffers.
        self.register_buffer('log_at', log_at.float())
        self.register_buffer('log_bt', log_bt.float())
        self.register_buffer('log_ct', log_ct.float())
        self.register_buffer('log_cumprod_at', log_cumprod_at.float())
        self.register_buffer('log_cumprod_bt', log_cumprod_bt.float())
        self.register_buffer('log_cumprod_ct', log_cumprod_ct.float())
        self.register_buffer('log_1_min_ct', log_1_min_ct.float())
        self.register_buffer('log_1_min_cumprod_ct', log_1_min_cumprod_ct.float())
        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps))
        
        self.zero_vector = None

        self.condition_encoder = ConditionEmbeddingModel(num_condition_types=self.num_condition_types)

    def multinomial_kl(self, log_prob1, log_prob2):   # compute KL loss on log_prob
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

    def q_pred_one_timestep(self, log_x_t, t, batch):         # q(xt|xt_1) [-2] mask_category [-1] mask_poi
        B,V,L=log_x_t.shape
        t = t.unsqueeze(1).repeat(1,L)
        log_x_start = rearrange(log_x_t, 'b v l -> b l v')

        log_x_start_category = log_x_start[batch.category_mask.bool()]
        t_tmp = t[batch.category_mask.bool()]

        selected_range = log_x_start_category[:,self.num_spectial:self.num_spectial+self.type_classes]

        log_ct1 = extract(self.log_ct1, t_tmp, selected_range.shape)         # ct~
        log_1_min_ct1 = extract(self.log_1_min_ct1, t_tmp, selected_range.shape)       # 1-ct~

        selected_range = selected_range + log_1_min_ct1
        log_x_start_category[:,self.num_spectial:self.num_spectial+self.type_classes] = selected_range
        log_x_start_category = torch.cat([log_x_start_category[:,:-2],log_add_exp(log_x_start_category[:,-2:-1],log_ct1), log_x_start_category[:,-1:]],dim=-1)
        log_x_start[batch.category_mask.bool()] = log_x_start_category

        log_x_start_poi = log_x_start[batch.poi_mask.bool()]
        selected_range = log_x_start_poi[:,self.num_spectial+self.type_classes:-2]
        t_tmp = t[batch.poi_mask.bool()]
        log_at = extract(self.log_at, t_tmp, selected_range.shape) 
        log_bt = extract(self.log_bt, t_tmp, selected_range.shape)             # bt
        log_ct = extract(self.log_ct, t_tmp, selected_range.shape)             # ct
        log_1_min_ct = extract(self.log_1_min_ct, t_tmp, selected_range.shape)          # 1-ct

        selected_range = log_add_exp(selected_range +log_at, log_bt)
        log_x_start_poi[:,self.num_spectial+self.type_classes:-2] = selected_range
        log_x_start_poi = torch.cat([log_x_start_poi[:,:-1],log_add_exp(log_x_start_poi[:,-1:]+log_1_min_ct, log_ct)],dim=-1)
        log_x_start[batch.poi_mask.bool()] = log_x_start_poi

        log_probs = rearrange(log_x_start, 'b l v -> b v l')

        return log_probs

    def q_pred(self, log_x_start, t, batch):           # q(xt|x0)
        t = (t + (self.num_timesteps + 1))%(self.num_timesteps + 1)
        B,V,L=log_x_start.shape
        t = t.unsqueeze(1).repeat(1,L)

        log_x_start = rearrange(log_x_start, 'b v l -> b l v')

        log_x_start_category = log_x_start[batch.category_mask.bool()]
        selected_range = log_x_start_category[:,self.num_spectial:self.num_spectial+self.type_classes]
        t_tmp = t[batch.category_mask.bool()]

        log_cumprod_ct1 = extract(self.log_cumprod_ct1, t_tmp, selected_range.shape)         # ct~
        log_1_min_cumprod_ct1 = extract(self.log_1_min_cumprod_ct1, t_tmp, selected_range.shape)       # 1-ct~


        selected_range = selected_range + log_1_min_cumprod_ct1
        log_x_start_category[:,self.num_spectial:self.num_spectial+self.type_classes] = selected_range
        log_x_start_category = torch.cat([log_x_start_category[:,:-2],log_add_exp(log_x_start_category[:,-2:-1],log_cumprod_ct1), log_x_start_category[:,-1:]],dim=-1)
        log_x_start[batch.category_mask.bool()] = log_x_start_category

        log_x_start_poi = log_x_start[batch.poi_mask.bool()]
        selected_range = log_x_start_poi[:,self.num_spectial+self.type_classes:-2]
        t_tmp = t[batch.poi_mask.bool()]

        log_cumprod_at = extract(self.log_cumprod_at, t_tmp, selected_range.shape)
        log_cumprod_bt = extract(self.log_cumprod_bt, t_tmp, selected_range.shape)         # bt~
        log_cumprod_ct = extract(self.log_cumprod_ct, t_tmp, selected_range.shape)         # ct~
        log_1_min_cumprod_ct = extract(self.log_1_min_cumprod_ct, t_tmp, selected_range.shape)       # 1-ct~

        selected_range = log_add_exp(selected_range +log_cumprod_at, log_cumprod_bt)
        log_x_start_poi[:,self.num_spectial+self.type_classes:-2] = selected_range
        log_x_start_poi = torch.cat([log_x_start_poi[:,:-1],log_add_exp(log_x_start_poi[:,-1:]+log_1_min_cumprod_ct, log_cumprod_ct)],dim=-1)
        log_x_start[batch.poi_mask.bool()] = log_x_start_poi


        log_probs = rearrange(log_x_start, 'b l v -> b v l')
            
        return log_probs

    def predict_start(self, log_x_t, cond_emb, t, batch):          # p(x0|xt)
        x_t = log_onehot_to_index(log_x_t)
        if self.amp == True:
            with autocast():
                out = self.transformer(x_t, cond_emb, t, batch)
        else:
            out = self.transformer(x_t, cond_emb, t, batch)
        assert out.size(0) == x_t.size(0)
        assert out.size(1) == self.num_classes-2
        assert out.size()[2:] == x_t.size()[1:]
        
        log_pred = F.log_softmax(out.double(), dim=1).float()
        batch_size = log_x_t.size()[0]

        zero_vector = torch.zeros(batch_size, 2, log_x_t.size(2)).type_as(log_x_t)- 70
        log_pred = torch.cat((log_pred, zero_vector), dim=1)
        log_pred = torch.clamp(log_pred, -70, 0)
        return log_pred
    
    def predict_start_with_truncate(self, log_x_t, cond_emb, t, batch, truncation_k=15):  # p(x0|xt)
        x_t = log_onehot_to_index(log_x_t)
        if self.amp == True:
            with autocast():
                out = self.transformer(x_t, cond_emb, t, batch)
        else:
            out = self.transformer(x_t, cond_emb, t, batch)
        assert out.size(0) == x_t.size(0)
        assert out.size(1) == self.num_classes-2
        assert out.size()[2:] == x_t.size()[1:]
        
        log_pred = F.log_softmax(out.double(), dim=1).float()

        val, ind = log_pred.topk(k=truncation_k, dim=1)
        probs = torch.full_like(log_pred, -70)
        log_pred = probs.scatter_(1, ind, val)

        batch_size = log_x_t.size()[0]
        zero_vector = torch.zeros(batch_size, 2, log_x_t.size(2)).type_as(log_x_t)- 70
        log_pred = torch.cat((log_pred, zero_vector), dim=1)
        log_pred = torch.clamp(log_pred, -70, 0)

        return log_pred

 
    
    def q_posterior(self, log_x_start, log_x_t, t, batch):            

        assert t.min().item() >= 0 and t.max().item() < self.num_timesteps
        log_EV_xtmin_given_xt_given_xstart = self.q_pred(log_x_start, t - 1, batch)
        return torch.clamp(log_EV_xtmin_given_xt_given_xstart, -70, 0)

    def p_pred(self, log_x, cond_emb, t, batch):             # if x0, first p(x0|xt), then sum(q(xt-1|xt,x0)*p(x0|xt))
        if self.parametrization == 'x0':
            log_x_recon = self.predict_start_with_truncate(log_x, cond_emb, t, batch)
            log_model_pred = self.q_posterior(
                log_x_start=log_x_recon, log_x_t=log_x, t=t, batch=batch)
        elif self.parametrization == 'direct':
            log_model_pred = self.predict_start(log_x, cond_emb, t, batch)
        else:
            raise ValueError
        return log_model_pred, log_x_recon

    @torch.no_grad()
    def p_sample(self, log_x, cond_emb,  t, batch):               # sample q(xt-1) for next step from  xt, actually is p(xt-1|xt)
        model_log_prob, log_x_recon = self.p_pred(log_x, cond_emb, t, batch)
        # Gumbel sample
        out = self.log_sample_categorical(model_log_prob)
        return out

    def log_sample_categorical(self, logits):           # use gumbel to sample onehot vector from log probability
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=1)
        log_sample = index_to_log_onehot(sample, self.num_classes)
        return log_sample

    def q_sample(self, log_x_start, t, batch):                 # diffusion step, q(xt|x0) and sample xt
        log_EV_qxt_x0 = self.q_pred(log_x_start, t, batch)
        log_sample = self.log_sample_categorical(log_EV_qxt_x0)

        return log_sample

    def sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()
            t = torch.multinomial(pt_all, num_samples=b, replacement=True)
            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == 'uniform':
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        else:
            raise ValueError

    @property
    def device(self):
        return self.transformer.to_logits[-1].weight.device ##todo

    def training_losses(
        self,
        batch,
        is_train=True
        ):
        b, device = batch.batch_size, batch.device
        assert self.loss_type == 'vb_stochastic'
        x_start = batch.checkin_sequences
        t, pt = self.sample_time(b, device, 'importance')
        log_x_start = index_to_log_onehot(x_start, self.num_classes)
        log_xt = self.q_sample(log_x_start=log_x_start, t=t, batch=batch) #gt x_t #use matrix

        cond_emb = self.condition_encoder(batch)

        log_x0_recon = self.predict_start(log_xt, cond_emb, t=t, batch=batch) 
        log_x0_recon = log_x0_recon.transpose(1, 2)

        losses = {}
        loss = F.cross_entropy(log_x0_recon.reshape(-1, log_x0_recon.shape[-1]), x_start.flatten(),ignore_index=3, reduce=False)
        loss = loss.reshape(x_start.size(0), -1)
        losses['loss'] = torch.mean(loss, -1)
        return losses['loss']

    def sample_fast(
            self,
            batch,
            content_token = None,
            **kwargs):
        B, L = batch.batch_size, batch.content_len

        device = self.log_at.device

        batch.device = device

        cond_emb = self.condition_encoder(batch)

        mask_poi=self.num_classes-1
        mask_cat=self.num_classes-2
        
        bottom=torch.tensor([2],device=device)
        input=torch.ones(B,L,dtype=torch.int64,device=device) *3 ##padding

        for i in range(B):
            seq_len = batch.unpadded_length[i]
            head=torch.tensor([0] + [mask_cat] * seq_len ,device=device)
            body=torch.tensor([1] + [mask_poi] * seq_len ,device=device)

            tmp=torch.cat([head,body,bottom],dim=-1)
            input[i][:len(tmp)]=tmp

        log_z = index_to_log_onehot(input,self.num_classes)
        start_step = self.num_timesteps
        with torch.no_grad():
            for diffusion_index in range(start_step - 1, -1, -1):
                t = torch.full((B,), diffusion_index, device=device, dtype=torch.long)
                log_z = self.p_sample(log_z, cond_emb, t, batch)  # log_z is log_onehot

        content_token = log_onehot_to_index(log_z)

        return Batch(
            time=batch.time,
            condition1=batch.condition1, 
            condition2=batch.condition2,
            condition3=batch.condition3,
            condition4=batch.condition4,
            condition5=batch.condition5,
            condition6=batch.condition6,
            condition1_indicator=batch.condition1_indicator,
            condition2_indicator=batch.condition2_indicator,
            condition3_indicator=batch.condition3_indicator,
            condition4_indicator=batch.condition4_indicator,
            condition5_indicator=batch.condition5_indicator,
            condition6_indicator=batch.condition6_indicator,
            mask=batch.mask, 
            tmax=batch.tmax,
            checkin_sequences=content_token,
            category_mask=batch.category_mask,
            poi_mask=batch.poi_mask,
            tau=batch.tau,
            unpadded_length=batch.unpadded_length
        )
