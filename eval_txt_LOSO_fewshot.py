import os
import torch
from models import models_txt as networks
from modules import GreedyCTCDecoder, AttrDict, RMSELoss
from utils import word_index
import torch.nn as nn
from NeuroTalkDataset import myDataset
import time
import torch.optim.lr_scheduler
import numpy as np
import torchaudio
from torchmetrics import CharErrorRate
import json
import argparse
import random
from train_txt_fewshot import train as eval
from eval_txt_all import save_test_all



def main(args):
    
    device = torch.device(f'cuda:{args.gpuNum[0]}' if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device) # change allocation of current GPU
    print ('Current cuda device: {} '.format(torch.cuda.current_device())) # check
    print('The number of available GPU:{}'.format(torch.cuda.device_count()))
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # define generator
    config_file = os.path.join(args.model_config, 'config_txt_g.json')
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h_g = AttrDict(json_config)
    model_g = networks.Generator(h_g).cuda()
    
    # define discriminator
    config_file = os.path.join(args.model_config, 'config_txt_d.json')
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h_d = AttrDict(json_config)
    model_d = networks.Discriminator(h_d).cuda()
    
    # STT Wav2Vec
    bundle = torchaudio.pipelines.HUBERT_ASR_LARGE
    model_STT = bundle.get_model().cuda()
    decoder_STT = GreedyCTCDecoder(labels=bundle.get_labels())
    args.word_index, args.word_length = word_index(args.word_label, bundle)
    
    # Parallel setting
    model_g = nn.DataParallel(model_g, device_ids=args.gpuNum)
    model_d = nn.DataParallel(model_d, device_ids=args.gpuNum)
    model_STT = nn.DataParallel(model_STT, device_ids=args.gpuNum)

    # loss function
    criterion_recon = nn.MSELoss().cuda()
    criterion_ctc = nn.CTCLoss().cuda()
    criterion_adv = nn.BCELoss().cuda()
    criterion_cl = nn.CrossEntropyLoss().cuda()
    RMSE = RMSELoss().cuda()
    CER = CharErrorRate().cuda()

    # Directory
    logDir = args.logDir + str(args.fewshot) + '-shot/'
    saveDir = logDir + 'LOSO_fineTune_sub' + str(args.sub) + '_' + args.task + args.comments
    args.savemodel = args.pretrain_model + '/savemodel'
    args.tunemodel = saveDir + '/tunemodel'
    
    if args.evalmodel == '/tunemodel':
        evalmodel = args.tunemodel
    else:
        evalmodel = args.savemodel
    

    # Load trained model
    loc_g = os.path.join(evalmodel, 'BEST_checkpoint_g.pt')
    loc_d = os.path.join(evalmodel, 'BEST_checkpoint_d.pt')

    if os.path.isfile(loc_g):
        print("=> loading fine-tuned checkpoint '{}'".format(loc_g))
        checkpoint_g = torch.load(loc_g, map_location='cpu')
        model_g.load_state_dict(checkpoint_g['state_dict'])
        print('Load {}th epoch model'.format(checkpoint_g['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(loc_g))
        return 0

    if os.path.isfile(loc_d):
        print("=> loading fine-tuned checkpoint '{}'".format(loc_d))
        checkpoint_d = torch.load(loc_d, map_location='cpu')
        model_d.load_state_dict(checkpoint_d['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(loc_d))
        return 0
    
    
   # create the directory if not exist
    if not os.path.exists(args.logDir):
        os.mkdir(args.logDir)
        
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)

    args.saveword = saveDir + '/epoword'
    if not os.path.exists(args.saveword):
        os.mkdir(args.saveword)


    # log save
    logFileLoc = args.saveword + '/eval_pred_words.txt'
    if os.path.isfile(logFileLoc):
        args.logger = open(logFileLoc, 'a')
    else:
        args.logger = open(logFileLoc, 'w')
        args.logger.write("%s\t%s\t%s"
                          %('idx', 'gt', 'pred'))
    args.logger.flush()


    # Data loader define
    generator = torch.Generator().manual_seed(args.seed)
     
    testset = myDataset(mode=1, data=args.dataLoc+'/sub%02d'%args.sub, task=args.task)  # file='./EEG_EC_Data_csv/train.txt'
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=4*len(args.gpuNum), pin_memory=True)
    
    epoch = 0
    start_time = time.time()
     
    print('Test')
    Te_losses = eval(args, test_loader, 
                       (model_g, model_d, model_STT, decoder_STT), 
                       (criterion_recon, criterion_ctc, criterion_adv, criterion_cl, RMSE, CER), 
                       ([],[]), 
                       epoch,
                       False, False)
    
    print('Save')
    save_test_all(args, test_loader, (model_g, model_d, model_STT, decoder_STT), Te_losses)
    
    time_taken = time.time() - start_time
    print("Time: %.2f\n"%time_taken)
    
    args.logger.close()


if __name__ == '__main__':
    
    dataLoc = './sampledata/'
    logDir = './logs/'
    
    pretrain_model = logDir + 'LOSO_pretrain'

    subjects = list(range(1,22))
    subNum = 1 # 0~4
    
    
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--model_config', type=str, default='./models', help='config for model folder path')
    parser.add_argument('--dataLoc', type=str, default=dataLoc)
    parser.add_argument('--pretrain_model', type=str, default=pretrain_model)
    parser.add_argument('--config', type=str, default='./config_txt_fewshot.json')
    parser.add_argument('--logDir', type=str, default=logDir)
    parser.add_argument('--resume', type=bool, default=True)
    parser.add_argument('--gpuNum', type=int, nargs='+', default=[0,1])
    parser.add_argument('--batch_size', type=int, default=10) 
    parser.add_argument('--fewshot', default=5) # All: -1
    parser.add_argument('--subjects', default=subjects)
    parser.add_argument('--subNum', type=int, default=subNum)
    parser.add_argument('--task', type=str, default='SpokenEEG')
    parser.add_argument('--evalmodel', type=str, default='/tunemodel') #tunemodel, savemodel
    parser.add_argument('--comments', type=str, default='')
    parser.add_argument('--unseen', type=str, default='stop')
    parser.add_argument('--save_epo', type=int, default=1)
    
    args = parser.parse_args()
    
    with open(args.config) as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)
    
    random.seed(args.seed)
    args.subjects = random.sample(subjects,5)
    args.sub = args.subjects[args.subNum]
    
    main(args)        
    
    
    