import argparse
import os
import sys
sys.path.append('..')
import pandas
from gains.model import SetTransformer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import pickle
import random
import sys
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils
from torch import Tensor
from torch.utils.data import DataLoader

from gains.model import PPO
from feature_env import FeatureEvaluator, base_path
from gains.train_utils import FSDataset_set_tf
from record import SelectionRecord
from utils.logger import info, error
from gains.train_set_TF import main as train_set_TF
from gains.train_set_TF import args as train_set_TF_args
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--new_gen', type=int, default=200)
parser.add_argument('--task_name', type=str, choices=['spectf', 'svmguide3', 'german_credit', 'spam_base',
                                                      'ionosphere', 'megawatt1', 'uci_credit_card', 'openml_618',
                                                      'openml_589', 'openml_616', 'openml_607', 'openml_620',
                                                      'openml_637',
                                                      'openml_586', 'uci_credit_card', 'higgs', 'ap_omentum_ovary','activity'
                                                      , 'mice_protein', 'coil-20', 'isolet', 'minist', 'minist_fashion'], default='spectf')

parser.add_argument('--gpu', type=int, default=0, help='used gpu')
parser.add_argument('--fe', type=str, choices=['+', '', '-'], default='-')
parser.add_argument('--top_k', type=int, default=100)
parser.add_argument('--gen_num', type=int, default=25)
parser.add_argument('--max_step_size', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=512)

parser.add_argument('--lr_actor', type=float, default=0.0003)
parser.add_argument('--lr_critic', type=float, default=0.001)
parser.add_argument('--eps_clip', type=float, default=0.2)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--search_step', type=int, default=1000)
parser.add_argument('--ppo_hidden_size', type=int, default=128)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--set_tf_arch', type=str, choices=['SAB','ISAB'],default='ISAB')
parser.add_argument('--set_tf_hidden_size', type=int,default=128)

args = parser.parse_args()


baseline_name = [
    'kbest',
    'mrmr',
    'lasso',
    'rfe',
    'lassonet',
    'gfs',
    'sarlfs',  # feature_env, N_STATES, N_ACTIONS, EPISODE=-1, EXPLORE_STEPS=30
    'marlfs',  # feature_env, N_STATES, N_ACTIONS, EPISODE=-1, EXPLORE_STEPS=30
    'rra',
    'mcdm',
    'gains'
]
def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6



def choice_to_onehot(choice, eos):
    new_choice = []
    eos_batches = choice.data.eq(eos)
    eos_batches = ~eos_batches.cpu()

    for i in range(eos_batches.shape[0]):
        onehot = torch.zeros(eos)
        feat_seq = choice[i][eos_batches[i]]
        onehot[feat_seq] = 1
        new_choice.append(onehot)
    return torch.stack(new_choice, dim=0)

def output_to_state(output, valid_input, eos):
    eos_batches = valid_input.data.eq(eos)
    output[eos_batches] = eos
    new_state = choice_to_onehot(output, eos)
    return new_state

def get_reward(state, fe, epoch, epoches):
    reward_list = torch.empty(state.size(0), 1)
    print(f'{epoch}/{epoches} trajectroy collecting....')
    for i in range(state.shape[0]):
        reward = fe.report_performance(state[i].numpy(),rp=False, flag='test')
        reward_list[i] = reward
    print(f'{epoch}/{epoches}trajectroy collected!!!')
    return reward_list
def ppo_search(queue,ppo,set_tf,feat_len, epoches, fe, reward_weight):
    ppo.train()
    set_tf.eval()
    for epoch in range(epoches):
        for i, sample in enumerate(queue):
            encoder_input = sample['input']
            performance = sample['target']

            # valid feature index sequence to one-hot vector
            state = choice_to_onehot(encoder_input,feat_len)
            # select action with policy
            action = ppo.select_action(state) # size: [batch_size, 1, hidden_embedding_size]

            with torch.no_grad():
                feat_emb, _ = set_tf(encoder_input.cuda(set_tf.gpu).float())
                new_feat_emb = action + feat_emb.squeeze()
                new_output = set_tf.infer(new_feat_emb.unsqueeze(1))

            new_state = output_to_state(new_output, encoder_input, feat_len)
            new_state_reward = get_reward(new_state, fe, epoch, epoches)
            reward = reward_weight * (new_state_reward - performance) + (1 - reward_weight) * performance

            ppo.buffer.rewards.append(reward)

        ppo.update()

        new_selection = []
        new_choice = []
        predict_step_size = 0

        while len(new_selection) < args.new_gen:
            predict_step_size += 1
            info('Generate new architectures with step size {:d}'.format(predict_step_size))
            new_record = generate_new_records(queue, ppo, set_tf, fe.ds_size)
            for choice in new_record:
                if choice.sum() <= 0:
                    error('insufficient selection')
                    continue
                record = SelectionRecord(choice.numpy(), -1)
                if record not in fe.records.r_list and record not in new_selection:
                    new_selection.append(record)
                    new_choice.append(choice)
                if len(new_selection) >= args.new_gen:
                    break
            info(f'{len(new_selection)} new choice generated now', )
            if predict_step_size > args.max_step_size:
                break
        info(f'build {len(new_selection)} new choice !!!')

        new_choice_pt = torch.stack(new_choice)

        best_selection_test = None
        best_optimal_test = -1000
        for s in new_selection:
            test_data = fe.generate_data(s.operation, 'test')
            test_result = fe.get_performance(test_data)
            if test_result > best_optimal_test:
                best_selection_test = s.operation
                best_optimal_test = test_result
                info(f'found best on test : {best_optimal_test}')

        test_p = fe.report_performance(best_selection_test, flag='test')

        save_path = f'{base_path}/history/{fe.task_name}/ppo/{args.ppo_hidden_size}_{args.eps_clip}_{args.reward_tradeoff}_{test_p * 100}'
        os.makedirs(save_path, exist_ok=True)
        opt_path_test = os.path.join(save_path, 'best-ppo-results.hdf')
        choice_path = os.path.join(save_path, 'ppo_generated_choice.pt')

        fe.generate_data(best_selection_test, 'train').to_hdf(opt_path_test, key='train')
        fe.generate_data(best_selection_test, 'test').to_hdf(opt_path_test, key='test')
        ps = []
        info('given overall validation')
        report_head = 'RAW\t'
        raw_test = pandas.read_hdf(f'{base_path}/history/{fe.task_name}.hdf', key='raw_test')
        ps.append('{:.2f}'.format(fe.get_performance(raw_test) * 100))

        for method in baseline_name:
            report_head += f'{method}\t'

            spe_test = pandas.read_hdf(f'{base_path}/history/{fe.task_name}.hdf', key=f'{method}_test')
            ps.append('{:.2f}'.format(fe.get_performance(spe_test) * 100))

        report_head += 'Ours_Test'
        report = ''
        print(report_head)
        for per in ps:
            report += f'{per}&\t'
        report += '{:.2f}&\t'.format(test_p * 100)
        print(report)


        ppo_save_path = os.path.join(save_path, 'ppo.model_dict')
        ppo.save(ppo_save_path)
        print("--------------------------------------------------------------------------------------------")
        print("saving model at : " + ppo_save_path)
        print("--------------------------------------------------------------------------------------------")
        torch.save(new_choice_pt, choice_path)
        info(f'save generated choice to {choice_path}')
    return None


def generate_new_records(queue,ppo, set_tf, feat_len):
    with torch.no_grad():
        # inference
        for i, sample in enumerate(queue):
            encoder_input = sample['input']

            # valid feature index sequence to one-hot vector
            state = choice_to_onehot(encoder_input, feat_len)
            # select action with policy
            action = ppo.select_action(state)  # size: [batch_size, 1, hidden_embedding_size]
            feat_emb, _ = set_tf(encoder_input.cuda(set_tf.gpu).float())
            new_feat_emb = action + feat_emb.squeeze()
            new_output = set_tf.infer(new_feat_emb.unsqueeze(1))
            new_records = output_to_state(new_output, encoder_input, feat_len)
    return new_records
def select_top_k(choice: Tensor, labels: Tensor, k:int) -> (Tensor, Tensor):
    values, indices = torch.topk(labels, k, dim=0)
    return choice[indices.squeeze()], labels[indices.squeeze()]


def main():
    if not torch.cuda.is_available():
        info('No GPU found!')
        sys.exit(1)
    # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpu)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.enabled = True
    cudnn.benchmark = False
    cudnn.deterministic = True
    device = int(args.gpu)
    info(f"Args = {args}")
    with open(f'{base_path}/history/{args.task_name}/new_fe.pkl', 'rb') as f:
        fe: FeatureEvaluator = pickle.load(f)
    set_tf = SetTransformer(fe.ds_size, fe.ds_size, fe.ds_size, args)
    set_tf.load_state_dict(torch.load(f'{base_path}/history/{args.task_name}/set_tf.model_dict'))
    set_tf = set_tf.cuda(device)

    ppo_agent = PPO(fe.ds_size, args.set_tf_hidden_size, args.ppo_hidden_size, args.lr_actor, args.lr_critic, args.gamma, args.search_step, args.eps_clip)
    # ppo_agent.load_state_dict(torch.load(f'{base_path}/history/{args.task_name}/ppo_0.1_0.7818925323269501.model_dict'))
    ppo_agent = ppo_agent.cuda(device)

    valid_choice, valid_labels = fe.get_record(0, eos=fe.ds_size)


    top_selection, top_performance = select_top_k(valid_choice, valid_labels, args.top_k)

    infer_dataset = FSDataset_set_tf(top_selection, top_performance, sos_id=fe.ds_size, eos_id=fe.ds_size)
    infer_queue = DataLoader(infer_dataset, batch_size=len(infer_dataset), shuffle=False,
                             pin_memory=True)


    flag = ppo_search(infer_queue, ppo_agent, set_tf, fe.ds_size, args.epoch, fe, args.reward_tradeoff)
    return flag
if __name__ == '__main__':

    task_name_list = ['urbansound8k']

    for task_name in task_name_list:
        train_set_TF_args.task_name = task_name
        args.task_name = task_name
        flag = main()