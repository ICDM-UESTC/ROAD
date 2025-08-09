import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt 

from models.ROAD import ROAD
from models.model import *
from utils.data_loader import *
from utils.data_util import load_data
import os
import datetime
from torch.distributions import Normal, kl_divergence

def parse_args():
    config_args = {
        'lr': 0.0005,
        'dropout_gat': 0.3,
        'dropout': 0.3,
        'cuda': 0,
        'epochs_gat': 3000,
        'epochs': 2000,
        'weight_decay_gat': 1e-5,
        'weight_decay': 0,
        'seed': 2025,
        'model': 'ROAD',
        'num-layers': 3,
        'dim': 256,
        'r_dim': 256,
        'k_w': 10,
        'k_h': 20,
        'n_heads': 2,
        'dataset': 'DB15K',
        'pre_trained': 0,
        'encoder': 0,
        'image_features': 1,
        'text_features': 1,
        'patience': 5,
        'eval_freq': 100,
        'lr_reduce_freq': 500,
        'gamma': 1.0,
        'bias': 1,
        'neg_num': 2,
        'neg_num_gat': 2,
        'alpha': 1e-2,
        'alpha_gat': 0.2,
        'out_channels': 32,
        'kernel_size': 3,
        'batch_size': 1024,
        'save': 1,
        'n_exp': 3,
        'mu': 0.0001,
        'mu_alpha':0.01,
        'mu_beta':0.01, 
        'img_dim': 256,
        'txt_dim': 256,
        'beta_s':1e-5,
        'beta_t':1e-5,
        'beta_i':1e-5,
        'lamda_conf':1e-5,
        'lamda_cl':1e-5,
        'begin': 0,
        'std': 0,
        'stage1_epochs':700
    }
    parser = argparse.ArgumentParser()
    for param, val in config_args.items():
        parser.add_argument(f"--{param}", default=val, type=type(val))
    args = parser.parse_args()
    return args

args = parse_args()
print(args)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
print(f'Using: {args.device}')
torch.cuda.set_device(args.cuda)
for k, v in list(vars(args).items()):
    print(str(k) + ':' + str(v))

entity2id, relation2id, img_features, text_features, train_data, val_data, test_data = load_data(args.dataset)
print("Training data {:04d}".format(len(train_data[0])))

corpus = ConvECorpus(args, train_data, val_data, test_data, entity2id, relation2id)

if args.image_features:
    args.img = F.normalize(torch.Tensor(img_features), p=2, dim=1)
if args.text_features:
    args.desp = F.normalize(torch.Tensor(text_features), p=2, dim=1)
args.entity2id = entity2id
args.relation2id = relation2id

model_name = {'ROAD': ROAD}
time.sleep(5)

def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)  
            if m.bias is not None:
                nn.init.zeros_(m.bias) 
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

def train_decoder(args):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    model = model_name[args.model](args)
    
    init_weights(model)
    
    args.img_dim = model.img_dim
    args.txt_dim = model.txt_dim
    print(str(model))
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.gamma)
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    print(f'Total number of parameters: {tot_params}')
    if args.cuda is not None and int(args.cuda) >= 0:
        model = model.to(args.device)
    corpus.batch_size = args.batch_size
    corpus.neg_num = args.neg_num


    t_total = time.time()
    best_val_metrics = model.init_metric_dict()
    best_test_metrics = model.init_metric_dict()
    training_range = tqdm(range(args.epochs))
    for epoch in training_range:
        model.train()
        epoch_loss = []

        t = time.time()
        corpus.shuffle()
        for batch_num in range(corpus.max_batch_num):
            optimizer.zero_grad()
            train_indices, train_values = corpus.get_batch(batch_num)
            _,train_values_new = corpus.get_batch(batch_num)
            train_indices = torch.LongTensor(train_indices)
            if args.cuda is not None and int(args.cuda) >= 0:
                train_indices = train_indices.to(args.device)
                train_values = train_values.to(args.device)
                train_values_new = train_values_new.to(args.device)
            output, embeddings, conf_pred= model.forward(train_indices)
            loss_s, loss_i, loss_t, loss_mm,loss_kl_s,loss_kl_i, loss_kl_t,cl_loss  = model.loss_func(output, train_values)
            
            pred_s ,pred_i, pred_t, pred_mm = output
            conf_s = torch.sum(pred_s*train_values_new, dim=1, keepdim=True)
            conf_i = torch.sum(pred_i*train_values_new, dim=1, keepdim=True)
            conf_t = torch.sum(pred_t*train_values_new, dim=1, keepdim=True)
            conf_mm = torch.sum(pred_mm*train_values_new, dim=1, keepdim=True)
                                
            loss_conf_raw = F.mse_loss(conf_pred[0], conf_s) \
             + F.mse_loss(conf_pred[1], conf_i) \
             + F.mse_loss(conf_pred[2], conf_t) \
             + F.mse_loss(conf_pred[3], conf_mm)
            if args.dataset == 'DB15K':
                lambda_conf_reg = 5e-4
            else:
                lambda_conf_reg = 1e-3
            conf_reg = 0.0
            for name, param in model.named_parameters():
                if 'modal_conf' in name and param.requires_grad:
                    conf_reg += torch.norm(param, p=2) ** 2
            loss_conf = loss_conf_raw + lambda_conf_reg * conf_reg
                    
            loss_bce =  loss_s + loss_i + loss_t + loss_mm 
            loss_kl = args.beta_s*loss_kl_s + args.beta_i * loss_kl_i + args.beta_t * loss_kl_t 
            loss = loss_bce + loss_kl + args.lamda_cl * cl_loss + args.lamda_conf * loss_conf

            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=0.5, norm_type=2)
            optimizer.step()

            epoch_loss.append(loss.data.item())


        training_range.set_postfix(loss=f"{np.sum(epoch_loss):.5f}")
        lr_scheduler.step()

        if (epoch + 1) % args.eval_freq == 0:
            print(f"Epoch {epoch + 1}: Evaluating on Test Set...")
            model.eval()
            with torch.no_grad():
                val_metrics, _  = corpus.get_validation_pred(model, 'test')   
            if val_metrics['MRR'] > best_test_metrics['MRR']:
                best_test_metrics['MRR'] = val_metrics['MRR']
            if val_metrics['MR'] < best_test_metrics['MR']:
                best_test_metrics['MR'] = val_metrics['MR']
            if val_metrics['Hits@1'] > best_test_metrics['Hits@1']:
                best_test_metrics['Hits@1'] = val_metrics['Hits@1']
            if val_metrics['Hits@3'] > best_test_metrics['Hits@3']:
                best_test_metrics['Hits@3'] = val_metrics['Hits@3']
            if val_metrics['Hits@10'] > best_test_metrics['Hits@10']:
                best_test_metrics['Hits@10'] = val_metrics['Hits@10']
            if val_metrics['Hits@100'] > best_test_metrics['Hits@100']:
                best_test_metrics['Hits@100'] = val_metrics['Hits@100']      
            print('\n'.join(['Epoch: {:04d}'.format(epoch + 1), model.format_metrics(val_metrics, 'test')]))
            print("\n\n")

            
    print('Total time elapsed: {:.4f}s'.format(time.time() - t_total))
    if not best_test_metrics:
        model.eval()
        with torch.no_grad():
            best_test_metrics, _ = corpus.get_validation_pred(model, 'test')
    print('\n'.join(['Test set results:', model.format_metrics(best_test_metrics, 'test')]))
    print("\n\n\n\n\n\n")

    if args.save:
        save_dir = f'./checkpoint/{args.dataset}/{current_time}'
        os.makedirs(save_dir, exist_ok=True) 
        torch.save(model.state_dict(), os.path.join(save_dir, f'{args.model}.pth'))
        print('Saved model!')

if __name__ == '__main__':
    train_decoder(args)
