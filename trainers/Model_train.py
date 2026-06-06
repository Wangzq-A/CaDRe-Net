import torch
from torch.nn import NLLLoss


def default_train(train_loader, model, optimizer, writer, iter_counter, args):
    way = model.way
    query_shot = model.shots[-1]
    target = torch.LongTensor([i // query_shot for i in range(query_shot * way)]).cuda()
    criterion = NLLLoss().cuda()

    lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('lr', lr, iter_counter)

    avg_loss = 0
    avg_acc = 0

    for i, (inp, _) in enumerate(train_loader):
        iter_counter += 1

        inp = inp.cuda()

        log_prediction, f_q, f_rec = model(inp)

        loss_cls = criterion(log_prediction, target)
        loss_rec = torch.mean((f_q - f_rec) ** 2)

        alpha = args.alpha
        loss_total = loss_cls + alpha * loss_rec

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        loss_value = loss_total.item()

        _, max_index = torch.max(log_prediction, 1)
        acc = 100 * torch.sum(torch.eq(max_index, target)).item() / (query_shot * way)

        avg_acc += acc
        avg_loss += loss_value

    avg_acc = avg_acc / (i + 1)
    avg_loss = avg_loss / (i + 1)

    writer.add_scalar('CaDRe_Net_loss', avg_loss, iter_counter)
    writer.add_scalar('train_acc', avg_acc, iter_counter)

    return iter_counter, avg_acc, avg_loss

def train_parser():
    parser = argparse.ArgumentParser()

    ## general hyper-parameters
    parser.add_argument("--opt", help="optimizer", choices=['adam','sgd'], default='sgd')
    parser.add_argument("--lr", help="initial learning rate", type=float, default=0.1)
    parser.add_argument("--gamma", help="learning rate cut scalar", type=float, default=0.1)
    parser.add_argument("--epoch", help="number of epochs before lr is cut by gamma", type=int, default=150)
    parser.add_argument("--weight_decay", help="weight decay for optimizer", type=float, default=5e-4)
    parser.add_argument('--gpu', default=2, type=int, help='gpu id')
    parser.add_argument("--seed", help="random seed", type=int, default=42)
    parser.add_argument("--val_epoch", help="number of epochs before eval on val", type=int, default=1)
    parser.add_argument("--resnet", help="whether use resnet12 as backbone or not", action="store_true")
    parser.add_argument("--nesterov", help="nesterov for sgd", action="store_true")
    parser.add_argument("--batch_size", help="batch size used during pre-training", type=int)
    parser.add_argument('--decay_epoch', nargs='+',help='epochs that cut lr', type=int)
    parser.add_argument("--pre", help="whether use pre-resized 84x84 images for val and test", action="store_true")
    parser.add_argument("--no_val", help="don't use validation set, just save model at final timestep", action="store_true")
    parser.add_argument("--train_way", help="training way", type=int)
    parser.add_argument("--test_way", help="test way", type=int, default=5)
    parser.add_argument("--train_shot", help="number of support images per class for meta-training and meta-testing during validation", type=int)
    parser.add_argument("--test_shot", nargs='+', help="number of support images per class for meta-testing during final test", type=int)
    parser.add_argument("--train_query_shot", help="number of query images per class during meta-training", type=int, default=15)
    parser.add_argument("--test_query_shot", help="number of query images per class during meta-testing", type=int, default=16)
    parser.add_argument("--train_transform_type", help="size transformation type during training", type=int)
    parser.add_argument("--test_transform_type", help="size transformation type during inference", type=int)
    parser.add_argument("--val_trial", help="number of meta-testing episodes during validation", type=int, default=1000)
    parser.add_argument("--detailed_name", help="whether include training details in the name", action="store_true")
    parser.add_argument("--dataset", choices=['stanford_dog', 'stanford_car', 'cub_raw', 'meta_iNat', 'tiered_meta_iNat'])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--model", choices=['C2_Net'], default='C2_Net')
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_epoch", type=int, default=0)

    args = parser.parse_args()

    return args
