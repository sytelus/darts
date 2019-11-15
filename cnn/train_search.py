import  os,sys,time, glob
import  numpy as np
import  torch
import  utils
import  logging
import  argparse
import  torch.nn as nn
from    torch import optim
import  torchvision.datasets as dset
import  torch.backends.cudnn as cudnn

from    model_search import Network
from    arch import Arch


# this file will find the architecture but will not fully train the model
# once architecture is found, use train.py to train the model from scratch


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batchsz', type=int, default=64, help='batch size')
parser.add_argument('--lr', type=float, default=0.025, help='init learning rate')
parser.add_argument('--lr_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for model parameters')
parser.add_argument('--wd', type=float, default=3e-4, help='weight decay for model parameters')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_ch', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_len', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--exp_path', type=str, default='search', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping range')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training/val splitting')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_lr', type=float, default=3e-4, help='learning rate for arch parameters')
parser.add_argument('--arch_wd', type=float, default=1e-3, help='weight decay for arch parameters')
args = parser.parse_args()

args.exp_path += str(args.gpu)
utils.create_exp_dir(args.exp_path, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.exp_path, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# available GPU IDs
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
# select first visible GPU ID
device = torch.device('cuda:0')

def main():
    np.random.seed(args.seed)
    # bechmark mode will cause cuDNN to evaluate algorithms for current machine and adapt to the best
    cudnn.benchmark = True
    cudnn.enabled = True
    torch.manual_seed(args.seed)

    # ================================================
    total, used = os.popen(
        'nvidia-smi --query-gpu=memory.total,memory.used --format=csv,nounits,noheader'
            ).read().split('\n')[args.gpu].split(',')
    total = int(total)
    used = int(used)

    print('Total GPU mem:', total, 'used:', used)


    # try:
    #     block_mem = 0.85 * (total - used)
    #     print(block_mem)
    #     x = torch.empty((256, 1024, int(block_mem))).cuda()
    #     del x
    # except RuntimeError as err:
    #     print(err)
    #     block_mem = 0.8 * (total - used)
    #     print(block_mem)
    #     x = torch.empty((256, 1024, int(block_mem))).cuda()
    #     del x
    #
    #
    # print('reuse mem now ...')
    # ================================================

    if not args.unrolled:
        print('WARNING: unrolled arg is NOT true. This is useful only for abalation study for bilevel optimization!')

    logging.info('GPU device = %d' % args.gpu)
    logging.info("args = %s", args)


    criterion = nn.CrossEntropyLoss().to(device) # CIFAR classification task
    # 16 inital channels, num_classes=10, 8 cells (layers)
    model = Network(args.init_ch, 10, args.layers, criterion).to(device)

    logging.info("Total param size = %f MB", utils.count_parameters_in_MB(model))

    # this is the optimizer to optimize
    optimizer = optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd)

    # note that we get only train set here and break it down in 1/2 to get validation set
    # cifar10 has 60K images in 10 classes, 50k in train, 10k in test
    # so ultimately we have 25K train, 25K val, 10k test
    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

    num_train = len(train_data) # 50000
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train)) # 25000

    # generate random batches of 64 on train/val subsets
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batchsz,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=2)
    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batchsz,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:]),
        pin_memory=True, num_workers=2)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, float(args.epochs), eta_min=args.lr_min)

    # arch is sort of meta model that would update theta and alpha parameters
    arch = Arch(model, args)

    # in this phase we only run 50 epochs
    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('\nEpoch: %d lr: %e', epoch, lr)

        # genotype extracts the highest weighted two primitives per node
        # this is for information dump only
        genotype = model.genotype()
        logging.info('Genotype: %s', genotype)

        # print(F.softmax(model.alphas_normal, dim=-1))
        # print(F.softmax(model.alphas_reduce, dim=-1))

        # training
        train_acc, train_obj = train(train_queue, valid_queue, model, arch, criterion, optimizer, lr)
        logging.info('train acc: %f', train_acc)

        # validation
        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid acc: %f', valid_acc)

        utils.save(model, os.path.join(args.exp_path, 'search.pt'))


def train(train_queue, valid_queue, model, arch, criterion, optimizer, lr):
    """

    :param train_queue: train loader
    :param valid_queue: validate loader
    :param model: network
    :param arch: Arch class
    :param criterion:
    :param optimizer:
    :param lr:
    :return:
    """
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    valid_iter = iter(valid_queue)

    for step, (x, target) in enumerate(train_queue):

        batchsz = x.size(0)
        model.train() # put model into train mode

        # [b, 3, 32, 32], [40]
        x, target = x.to(device), target.cuda(non_blocking=True)
        x_search, target_search = next(valid_iter) # [b, 3, 32, 32], [b]
        x_search, target_search = x_search.to(device), target_search.cuda(non_blocking=True)

        # 1. update alpha
        arch.step(x, target, x_search, target_search, lr, optimizer, unrolled=args.unrolled)

        logits = model(x)
        loss = criterion(logits, target)

        # 2. update weight
        optimizer.zero_grad()
        loss.backward()
        # apparently gradient clipping is important
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        # as our arch parameters (i.e. alpha) is kept seperate, they don't get updated
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        losses.update(loss.item(), batchsz)
        top1.update(prec1.item(), batchsz)
        top5.update(prec5.item(), batchsz)

        if step % args.report_freq == 0:
            logging.info('Step:%03d loss:%f acc1:%f acc5:%f', step, losses.avg, top1.avg, top5.avg)

    return top1.avg, losses.avg


def infer(valid_queue, model, criterion):
    """
    For a given model we just evaluate metrics on validation set.
    Note that this model is not final, i.e., each node i has i+2 edges 
    and each edge with 8 primitives and associated wieghts.

    :param valid_queue:
    :param model:
    :param criterion:
    :return:
    """
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, (x, target) in enumerate(valid_queue):

            x, target = x.to(device), target.cuda(non_blocking=True)
            batchsz = x.size(0)

            logits = model(x)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            losses.update(loss.item(), batchsz)
            top1.update(prec1.item(), batchsz)
            top5.update(prec5.item(), batchsz)

            if step % args.report_freq == 0:
                logging.info('>> Validation: %3d %e %f %f', step, losses.avg, top1.avg, top5.avg)

    return top1.avg, losses.avg


if __name__ == '__main__':
    main()
