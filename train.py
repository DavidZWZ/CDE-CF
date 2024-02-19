import utils
import torch
import numpy as np
import dataloader
from parse import parse_args
import multiprocessing
import os
from os.path import join
from model import LightGCN, PureMF, UltraGCN, SGCN, LTOCF, LayerGCN, IMP_GCN, EASE, NGCF, ODE_CF, CDE_CF
from trainers import GraphRecTrainer
from utils import EarlyStopping
from time import perf_counter
import pandas as pd

def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()

ROOT_PATH = "./"
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'runs')
FILE_PATH = join(CODE_PATH, 'checkpoints')
import sys
sys.path.append(join(CODE_PATH, 'sources')) 


if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)

args.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
args.cores = multiprocessing.cpu_count() // 2

utils.set_seed(args.seed)

# let pandas shut up
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)



#Recmodel = register.MODELS[world.model_name](world.config, dataset)
dataset = dataloader.Loader(args)
if args.model_name == 'LightGCN':
    model = LightGCN(args, dataset)
elif args.model_name == 'UltraGCN':
    constraint_mat = dataset.getConstraintMat()
    ii_neighbor_mat, ii_constraint_mat = dataset.get_ii_constraint_mat()
    model = UltraGCN(args, dataset, constraint_mat, ii_constraint_mat, ii_neighbor_mat)
elif args.model_name == 'SGCN':
    model = SGCN(args, dataset)
elif args.model_name =='LTOCF':
    model = LTOCF(args, dataset)
elif args.model_name == 'layerGCN':
    model = LayerGCN(args, dataset)
elif args.model_name == 'IMP_GCN':
    model = IMP_GCN(args, dataset)
elif args.model_name =='EASE':
    model = EASE(args, dataset)
elif args.model_name == 'NGCF':
    model = NGCF(args, dataset)
elif args.model_name =='ODE_CF':
    model = ODE_CF(args, dataset)
elif args.model_name =='CDE_CF':
    model = CDE_CF(args, dataset)
else:
    model = PureMF(args, dataset)
model = model.to(args.device)
trainer = GraphRecTrainer(model, dataset, args)

checkpoint_path = utils.getFileName("./checkpoints/", args)
print(f"load and save to {checkpoint_path}")

t = perf_counter()
if args.do_eval:
    #trainer.load(checkpoint_path)
    trainer.model.load_state_dict(torch.load(checkpoint_path))
    print(f'Load model from {checkpoint_path} for test!')
    scores, result_info, _ = trainer.complicated_eval()
else:
    val_result = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    early_stopping = EarlyStopping(checkpoint_path, patience=50, verbose=True)
    if args.model_name == 'layerGCN':
            trainer.model.pre_epoch_processing()
    for epoch in range(args.epochs):
        trainer.train(epoch)
        # evaluate on MRR
        if (epoch+1) %10==0:
            scores, _, _ = trainer.valid(epoch, full_sort=True)
            val_result.append(scores)
            early_stopping(np.array(scores[-1:]), trainer.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    print('---------------Change to Final testing!-------------------')
    # load the best model
    trainer.model.load_state_dict(torch.load(checkpoint_path))
    valid_scores, val_r, _ = trainer.valid('best', full_sort=True)
    scores, result_info, _ = trainer.test('best', full_sort=True)
    val_result.append(scores)

col_name = ["HIT@1", "NDCG@1", "HIT@5", "NDCG@5", "HIT@10", "NDCG@10", "HIT@15", "NDCG@15", "HIT@20", "NDCG@20", "HIT@40", "NDCG@40", "MRR"]
val_result = pd.DataFrame(val_result, columns=col_name)
path = './results/' +args.model_name + '_' + args.data_name+ '_val_result_t=' + str(args.t) +'_solver_' + args.solver  + '.csv'
val_result.to_csv(path, index=False)
train_time = perf_counter()-t
with open('./results/overall_time.txt', 'a') as f:
    f.writelines(path + ': '+ "Train time: {:.4f}s".format(train_time) + '\n')
    f.writelines('best_val: ' + val_r+'\n')
    f.writelines('test: ' + result_info+'\n')
    f.writelines('\n')
print("Train time: {:.4f}s".format(train_time))