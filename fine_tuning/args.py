
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--prefix', default=None,
                    type=str, help='prefix for model id')
parser.add_argument('--dataset', default='CUB_200_2011',
                    type=str, help='dataset')
parser.add_argument('--seed', default=None, type=int,
                    help='random seed (default: None, i.e., not fix the randomness).')
parser.add_argument('--model', default='ResNet50',
                    type=str, help='network structure')
parser.add_argument('--batch_size', default=32, type=int,
                    help='batch_size.')
parser.add_argument('--test_batch_size', default=100, type=int,
                    help='test_batch_size.')
parser.add_argument('--wd_rate', default=1e-4, type=float,
                    help='wd_rate.')
parser.add_argument('--use_cuda', default=0, type=int,
                    help='use_cuda device. -1 cpu.')
parser.add_argument('--num_epoch', default=40, type=int,
                    help='num_epoch.')
parser.add_argument('--fc_reinit', default=0, type=int,
                    help='>=1 if want fc_reinit')
parser.add_argument('--snapshot', default=100, type=int, help='snapshot epoch to save')
parser.add_argument('--outdir', default='outdir',
                    type=str, help='outdir')
parser.add_argument('--cyclic_num', type = int, default = 2)
parser.add_argument('--lr_init', type = float, default = 0.01)
parser.add_argument('--pretrained_model', default='/mnt/scratch/xiaoxiang/haozhe/icml2020_rifle/RIFLE/pretrained_models/ResNet50_pretrained',
                    type=str, help='pretrained model pathname')

args = parser.parse_args()

