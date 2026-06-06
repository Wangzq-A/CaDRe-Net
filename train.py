import os
from functools import partial
import Model_train
import eval
from datasets import dataloaders
from utils.util import *
from models.model import CaDReNet
from models.backbones.ResNet import ResNet12
from models.module.PCM import PCM
from models.module.MPPG import MPPG

args = Model_train.train_parser()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

fewshot_path = dataset_path(args)
pm = Model_train.Path_Manager(fewshot_path=fewshot_path, args=args)

train_loader = dataloaders.meta_train_dataloader(
    data_path=pm.train,
    way=args.train_way,
    shots=[args.train_shot, args.train_query_shot],
    transform_type=args.train_transform_type
)

args.save_folder = get_save_path(args)

train_func = partial(train.default_train, train_loader=train_loader)
tm = trainer.Train_Manager(args, path_manager=pm, train_func=train_func)
ev = eval.Eval_Manager(args, path_manager=pm, train_func=train_func)


def build_cadre_model(args):
    backbone = ResNet12(drop=True, drop_rate=0.1)
    pcm = PCM(num_classes=args.train_way)
    mppg = MPPG(in_channels=640)
  
    model = CaDReNet(
        backbone=backbone,
        pcm=pcm,
        mppg=mppg,
        way=args.train_way,
        shots=[args.train_shot, args.train_query_shot]
    )
    return model.cuda()

model = build_cadre_model(args) 

if args.resume:
    model = load_resume_point(args, model)

tm.train(model)
ev.evaluate(model)
