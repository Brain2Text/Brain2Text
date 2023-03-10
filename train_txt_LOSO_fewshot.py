import os
import torch
from models import models_txt as networks
from modules import GreedyCTCDecoder, AttrDict, RMSELoss, save_checkpoint
from utils import word_index
import torch.nn as nn
from NeuroTalkDataset_LOSO import myDataset
import time
import torch.optim.lr_scheduler
import numpy as np
import torchaudio
from torchmetrics import CharErrorRate
import json
import argparse
from torch.utils.tensorboard import SummaryWriter
import random
from train_txt_fewshot import train_G, train_D

    
def train(args, train_loader, models, criterions, optimizers, epoch, trainValid=True, logValid=True):
    '''
    :param args: general arguments
    :param train_loader: loaded for training/validation/test dataset
    :param model: model
    :param criterion: loss function
    :param optimizer: optimization algo, such as ADAM or SGD
    :param epoch: epoch number
    :return: losses
    '''
    
    (optimizer_g, optimizer_d) = optimizers
    
    # switch to train mode
    assert type(models) == tuple, "More than two models should be inputed (generator and discriminator)"
    
    epoch_loss_g = []
    epoch_loss_d = []
    
    epoch_acc_g = []
    epoch_acc_d = []
    
    epoch_loss_g_ns = []
    epoch_loss_d_ns = []
    
    epoch_acc_g_ns = []
    epoch_acc_d_ns = []
    
    args.save_data = True
    
    total_batches = len(train_loader)
    
    for i, (input, target_cl, voice, data_info) in enumerate(train_loader):    

        print("\rBatch [%5d / %5d]"%(i,total_batches), sep=' ', end='', flush=True)
        
        # subject append
        input = torch.reshape(input, (input.shape[0]*input.shape[1],input.shape[2],input.shape[3]))
        target_cl = torch.reshape(target_cl, (target_cl.shape[0]*target_cl.shape[1],target_cl.shape[2]))
        voice = torch.reshape(voice, (voice.shape[0]*voice.shape[1],voice.shape[2]))
        
        input = input.cuda()
        target_cl = target_cl.cuda()
        voice = voice.cuda()
        labels = torch.argmax(target_cl,dim=1) 
        
        # extract unseen
        idx_unseen=[]
        idx_seen=[]
        for j in range(len(labels)):
            if args.classname[labels[j]] == args.unseen:
                idx_unseen.append(j)
            else:
                idx_seen.append(j)
        
        input_ns = input[idx_unseen]
        target_cl_ns = target_cl[idx_unseen]
        voice_ns = voice[idx_unseen]
        labels_ns = labels[idx_unseen]
        
        input = input[idx_seen]
        target_cl = target_cl[idx_seen]
        voice = voice[idx_seen]
        labels = labels[idx_seen]
        
        
        # general training         
        if len(input) != 0:
            # train generator
            emission_recon, e_loss_g, e_acc_g = train_G(args, 
                                                        input, voice, labels,  
                                                        models, criterions, optimizer_g, 
                                                        epoch, trainValid)
            epoch_loss_g.append(e_loss_g)
            epoch_acc_g.append(e_acc_g)
        
            # train discriminator
            e_loss_d, e_acc_d = train_D(args, 
                                        emission_recon, voice, target_cl, labels,
                                        models, criterions, optimizer_d, 
                                        trainValid)
            epoch_loss_d.append(e_loss_d)
            epoch_acc_d.append(e_acc_d)
        
        # Unseen words training
        if len(input_ns) != 0 :
            # Unseen train generator
            emission_recon_ns, e_loss_g_ns, e_acc_g_ns = train_G(args, 
                                                                input_ns, voice_ns, labels_ns, 
                                                                models, criterions, optimizer_g, 
                                                                epoch, False)
            epoch_loss_g_ns.append(e_loss_g_ns)
            epoch_acc_g_ns.append(e_acc_g_ns)
            
            # Unseen train discriminator
            e_loss_d_ns, e_acc_d_ns = train_D(args, 
                                              emission_recon_ns, voice_ns, target_cl_ns, labels_ns, 
                                              models, criterions, optimizer_d, 
                                              False)
            epoch_loss_d_ns.append(e_loss_d_ns)
            epoch_acc_d_ns.append(e_acc_d_ns)
            
    epoch_loss_g = np.array(epoch_loss_g)
    epoch_acc_g = np.array(epoch_acc_g)
    epoch_loss_d = np.array(epoch_loss_d)
    epoch_acc_d = np.array(epoch_acc_d)
    
    epoch_loss_g_ns = np.array(epoch_loss_g_ns)
    epoch_acc_g_ns = np.array(epoch_acc_g_ns)
    epoch_loss_d_ns = np.array(epoch_loss_d_ns)
    epoch_acc_d_ns = np.array(epoch_acc_d_ns)
    
    
    args.loss_g = sum(epoch_loss_g[:,0]) / len(epoch_loss_g[:,0])
    args.loss_g_recon = sum(epoch_loss_g[:,1]) / len(epoch_loss_g[:,1])
    args.loss_g_valid = sum(epoch_loss_g[:,2]) / len(epoch_loss_g[:,2])
    args.loss_g_ctc = sum(epoch_loss_g[:,3]) / len(epoch_loss_g[:,3])
    args.rmse = sum(epoch_loss_g[:,4]) / len(epoch_loss_g[:,4])
    args.acc_g_valid = sum(epoch_acc_g[:,0]) / len(epoch_acc_g[:,0])
    args.cer_gt = sum(epoch_acc_g[:,1]) / len(epoch_acc_g[:,1])
    args.cer_recon = sum(epoch_acc_g[:,2]) / len(epoch_acc_g[:,2])
    args.acc_g_cl = sum(epoch_acc_g[:,3]) / len(epoch_acc_g[:,3])

    
    args.loss_d = sum(epoch_loss_d[:,0]) / len(epoch_loss_d[:,0])
    args.loss_d_valid = sum(epoch_loss_d[:,1]) / len(epoch_loss_d[:,1])
    args.loss_d_cl = sum(epoch_loss_d[:,2]) / len(epoch_loss_d[:,2])
    args.loss_d_real = sum(epoch_loss_d[:,3]) / len(epoch_loss_d[:,3])
    args.loss_d_fake = sum(epoch_loss_d[:,4]) / len(epoch_loss_d[:,4])
    args.acc_d_real = sum(epoch_acc_d[:,0]) / len(epoch_acc_d[:,0])
    args.acc_d_fake = sum(epoch_acc_d[:,1]) / len(epoch_acc_d[:,1])
    args.acc_cl_real = sum(epoch_acc_d[:,2]) / len(epoch_acc_d[:,2])
    args.acc_cl_fake = sum(epoch_acc_d[:,3]) / len(epoch_acc_d[:,3])
    
    # Unseen
    args.loss_g_ns = sum(epoch_loss_g_ns[:,0]) / len(epoch_loss_g_ns[:,0])
    args.loss_g_recon_ns = sum(epoch_loss_g_ns[:,1]) / len(epoch_loss_g_ns[:,1])
    args.loss_g_valid_ns = sum(epoch_loss_g_ns[:,2]) / len(epoch_loss_g_ns[:,2])
    args.loss_g_ctc_ns = sum(epoch_loss_g_ns[:,3]) / len(epoch_loss_g_ns[:,3])
    args.rmse_ns = sum(epoch_loss_g_ns[:,4]) / len(epoch_loss_g_ns[:,4])
    args.acc_g_valid_ns = sum(epoch_acc_g_ns[:,0]) / len(epoch_acc_g_ns[:,0])
    args.cer_gt_ns = sum(epoch_acc_g_ns[:,1]) / len(epoch_acc_g_ns[:,1])
    args.cer_recon_ns = sum(epoch_acc_g_ns[:,2]) / len(epoch_acc_g_ns[:,2])
    args.acc_g_cl_ns = sum(epoch_acc_g_ns[:,3]) / len(epoch_acc_g_ns[:,3])
    
    args.loss_d_ns = sum(epoch_loss_d_ns[:,0]) / len(epoch_loss_d_ns[:,0])
    args.loss_d_valid_ns = sum(epoch_loss_d_ns[:,1]) / len(epoch_loss_d_ns[:,1])
    args.loss_d_cl_ns = sum(epoch_loss_d_ns[:,2]) / len(epoch_loss_d_ns[:,2])
    args.loss_d_real_ns = sum(epoch_loss_d_ns[:,3]) / len(epoch_loss_d_ns[:,3])
    args.loss_d_fake_ns = sum(epoch_loss_d_ns[:,4]) / len(epoch_loss_d_ns[:,4])
    args.acc_d_real_ns = sum(epoch_acc_d_ns[:,0]) / len(epoch_acc_d_ns[:,0])
    args.acc_d_fake_ns = sum(epoch_acc_d_ns[:,1]) / len(epoch_acc_d_ns[:,1])
    args.acc_cl_real_ns = sum(epoch_acc_d_ns[:,2]) / len(epoch_acc_d_ns[:,2])
    args.acc_cl_fake_ns = sum(epoch_acc_d_ns[:,3]) / len(epoch_acc_d_ns[:,3])
    
   
    # tensorboard
    if logValid:
        if trainValid:
            tag = 'train'
        else:
            tag = 'valid'
            
        args.writer.add_scalar("Loss_G/{}".format(tag), args.loss_g, epoch)
        args.writer.add_scalar("CER/{}".format(tag), args.cer_recon, epoch)
        args.writer.add_scalar("ACC_G/{}".format(tag), args.acc_g_cl, epoch)
        
        args.writer.add_scalar("Loss_G_recon/{}".format(tag), args.loss_g_recon, epoch)
        args.writer.add_scalar("Loss_G_valid/{}".format(tag), args.loss_g_valid, epoch)
        args.writer.add_scalar("Loss_G_ctc/{}".format(tag), args.loss_g_ctc, epoch)
        
        args.writer.add_scalar("Loss_D_real/{}".format(tag), args.loss_d_real, epoch)
        args.writer.add_scalar("Loss_D_fake/{}".format(tag), args.loss_d_fake, epoch)
        
        # unseen
        args.writer.add_scalar("Loss_G_unseen/{}".format(tag), args.loss_g_ns, epoch)
        args.writer.add_scalar("CER_unseen/{}".format(tag), args.cer_recon_ns, epoch)
    
    print('\n[Seen] G_valid: %.4f D_R: %.4f / RMSE: %.4f / \nCER-gt: %.4f CER-recon: %.4f / g-cl: %.4f  / \ng-RMSE: %.4f g-lossValid: %.4f g-lossCTC: %.4f' 
          % (
             args.acc_g_valid, args.acc_d_real, 
             args.rmse, args.cer_gt, args.cer_recon, args.acc_g_cl,
             args.loss_g_recon, args.loss_g_valid, args.loss_g_ctc))
    
    print('\n[Unseen] G_valid: %.4f D_R: %.4f / RMSE: %.4f / \nCER-gt: %.4f CER-recon: %.4f / g-cl: %.4f  / \ng-RMSE: %.4f g-lossValid: %.4f g-lossCTC: %.4f' 
          % (
             args.acc_g_valid_ns, args.acc_d_real_ns, 
             args.rmse_ns, args.cer_gt_ns, args.cer_recon_ns, args.acc_g_cl_ns,
             args.loss_g_recon_ns, args.loss_g_valid_ns, args.loss_g_ctc_ns))
      
    return (args.loss_g, args.loss_g_recon, args.loss_g_valid, args.loss_g_ctc, args.acc_g_valid, args.cer_gt, args.cer_recon, args.loss_d, args.acc_d_real, args.acc_d_fake, args.acc_d_fake)



   
def saveDataOnly(args, transcript_recon, labels, epoch):
    
    str_tar, str_pred = 'None', 'None'
    
    str_tar = args.word_label[labels[0].item()].replace("|", ",")
    str_tar = str_tar.replace(" ", ",")
    
    str_pred = transcript_recon[0].replace("|", ",")
    str_pred = str_pred.replace(" ", ",")
    
    args.logger.write("\n%d\t%s\t%s"
                      %(epoch, str_tar, str_pred))
    
    args.logger.flush()


def main(args):
    
    device = torch.device(f'cuda:{args.gpuNum[0]}' if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device) # change allocation of current GPU
    print ('Current cuda device ', torch.cuda.current_device()) # check
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

    # optimizer
    optimizer_g = torch.optim.AdamW(model_g.parameters(), lr=args.lr_g, betas=(0.8, 0.99), weight_decay=0.01)
    optimizer_d = torch.optim.AdamW(model_d.parameters(), lr=args.lr_d, betas=(0.8, 0.99), weight_decay=0.01)

    scheduler_g = torch.optim.lr_scheduler.CyclicLR(optimizer_g, base_lr=args.lr_g/2, max_lr=args.lr_g*2, 
                                                    step_size_up=10, step_size_down=None, 
                                                    mode='triangular2', gamma=args.lr_g_decay, 
                                                    scale_fn=None, scale_mode='cycle', cycle_momentum=False, 
                                                    base_momentum=0.8, max_momentum=0.9, last_epoch=- 1, verbose=False)
    scheduler_d = torch.optim.lr_scheduler.CyclicLR(optimizer_d, base_lr=args.lr_d/2, max_lr=args.lr_d*2, 
                                                    step_size_up=10, step_size_down=None, 
                                                    mode='triangular2', gamma=args.lr_d_decay, 
                                                    scale_fn=None, scale_mode='cycle', cycle_momentum=False, 
                                                    base_momentum=0.8, max_momentum=0.9, last_epoch=- 1, verbose=False)


    # create the directory if not exist
    if not os.path.exists(logDir):
        os.mkdir(logDir)
     
    saveDir = args.logDir + 'LOSO_' + args.task + args.comments
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)

    args.saveword = saveDir + '/epoword'
    if not os.path.exists(args.saveword):
        os.mkdir(args.saveword)

    args.savemodel = saveDir + '/savemodel'
    if not os.path.exists(args.savemodel):
        os.mkdir(args.savemodel)

    args.logs = saveDir + '/logs'
    if not os.path.exists(args.logs):
        os.mkdir(args.logs)
        
    args.args = saveDir + '/args'
    if not os.path.exists(args.args):
        os.mkdir(args.args)
    

    # Load trained model
    start_epoch = 0
    start_best = 1000
    
    if args.pretrain:
        loc_g = os.path.join(args.trained_model, 'checkpoint_g.pt')
        loc_d = os.path.join(args.trained_model, 'checkpoint_d.pt')

        if os.path.isfile(loc_g):
            print("=> loading checkpoint '{}'".format(loc_g))
            checkpoint_g = torch.load(loc_g, map_location='cpu')
            model_g.load_state_dict(checkpoint_g['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(loc_g))

        if os.path.isfile(loc_d):
            print("=> loading checkpoint '{}'".format(loc_d))
            checkpoint_d = torch.load(loc_d, map_location='cpu')
            model_d.load_state_dict(checkpoint_d['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(loc_d))

    if args.resume:
        loc_g = os.path.join(args.savemodel, 'checkpoint_g.pt')
        loc_d = os.path.join(args.savemodel, 'checkpoint_d.pt')

        if os.path.isfile(loc_g):
            print("=> loading checkpoint '{}'".format(loc_g))
            checkpoint_g = torch.load(loc_g, map_location='cpu')
            model_g.load_state_dict(checkpoint_g['state_dict'])
            start_epoch = checkpoint_g['epoch'] + 1
            start_best = checkpoint_g['best_loss']
        else:
            print("=> no checkpoint found at '{}'".format(loc_g))

        if os.path.isfile(loc_d):
            print("=> loading checkpoint '{}'".format(loc_d))
            checkpoint_d = torch.load(loc_d, map_location='cpu')
            model_d.load_state_dict(checkpoint_d['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(loc_d))


    # log save
    logFileLoc = args.saveword + '/pred_words.txt'
    if os.path.isfile(logFileLoc):
        args.logger = open(logFileLoc, 'a')
    else:
        args.logger = open(logFileLoc, 'w')
        args.logger.write("%s\t%s\t%s"
                          %('Epoch', 'gt', 'pred'))
    args.logger.flush()
    
    # arguments save
    argsFileLoc = args.args + '/arguments.txt'
    argsave = open(argsFileLoc, 'w')
    argsave.write("eval_sub: %s \ntask: %s \nmax_epochs: %s \
                  \nl_g: %s \nlr_g: %s \nlr_g_decay: %s\
                      \nl_d: %s \nlr_g: %s \nlr_d_decay: %s\
                          \nbatch size: %s \nunseen word: %s\
                              \ndata Loc: %s \nlog Loc: %s"
                      %(args.eval_sub, args.task, args.max_epochs, 
                        args.l_g, args.lr_g, args.lr_g_decay, 
                        args.l_d, args.lr_d, args.lr_d_decay, 
                        args.batch_size, args.unseen,
                        args.dataLoc, args.logDir))
    argsave.flush()
        
        
    # Tensorboard setting
    args.writer = SummaryWriter(args.logs)
    
    # leave one subject out
    tr_sub = []
    # te_sub = [args.subjects[args.eval_sub]]
    for sub_ind in range(len(args.subjects)):
        if sub_ind not in args.eval_sub:
            tr_sub.append(args.subjects[sub_ind])
    
    # Data loader define
    generator = torch.Generator().manual_seed(args.seed)

    trainset = myDataset(mode=0, data=args.dataLoc, subjects = tr_sub, task=args.task)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=4*len(args.gpuNum), pin_memory=True)
    
    valset = myDataset(mode=2, data=args.dataLoc, subjects = tr_sub, task=args.task)
    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=4*len(args.gpuNum), pin_memory=True)

    epoch = start_epoch
    lr_g = 0
    lr_d = 0
    best_loss = start_best
    is_best = False
    epochs_since_improvement = 0
    
    for epoch in range(start_epoch, args.max_epochs):
        
        start_time = time.time()
        
        for param_group in optimizer_g.param_groups:
            lr_g = param_group['lr']
        for param_group in optimizer_d.param_groups:
            lr_d = param_group['lr']

        scheduler_g.step(epoch)
        scheduler_d.step(epoch)

        print("Epoch : %d/%d" %(epoch, args.max_epochs) )
        print("Learning rate for G: %.9f" %lr_g)
        print("Learning rate for D: %.9f" %lr_d)

        print("\nTrain")
        Tr_losses = train(args, train_loader, 
                          (model_g, model_d, model_STT, decoder_STT), 
                          (criterion_recon, criterion_ctc, criterion_adv, criterion_cl, RMSE, CER), 
                          (optimizer_g, optimizer_d), 
                          epoch,
                          True) 
        print("\nValid")
        Val_losses = train(args, val_loader, 
                           (model_g, model_d, model_STT, decoder_STT), 
                           (criterion_recon, criterion_ctc, criterion_adv, criterion_cl, RMSE, CER), 
                           ([],[]), 
                           epoch,
                           False)

        # Did validation loss improve?
        loss_total =  Val_losses[6]
        is_best = loss_total < best_loss
        best_loss = min(loss_total, best_loss)

        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        state_g = {'arch': str(model_g),
                 'state_dict': model_g.state_dict(),
                 'epoch': epoch,
                 'optimizer_state_dict': optimizer_g.state_dict(),
                 'best_loss': best_loss}
        
        state_d = {'arch': str(model_d),
                 'state_dict': model_d.state_dict(),
                 'epoch': epoch,
                 'optimizer_state_dict': optimizer_d.state_dict(),
                 'best_loss': best_loss}
        
        save_checkpoint(state_g, is_best, args.savemodel, 'checkpoint_g.pt')
        save_checkpoint(state_d, is_best, args.savemodel, 'checkpoint_d.pt')

        time_taken = time.time() - start_time
        print("Time: %.2f\n"%time_taken)
    
    args.writer.flush()
    args.logger.close()




if __name__ == '__main__':

    dataLoc = './sampledata/'
    logDir = './logs/'
    
    subjects = list(range(1,22))

    
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--model_config', type=str, default='./models', help='config for G & D folder path')
    parser.add_argument('--dataLoc', type=str, default=dataLoc)
    parser.add_argument('--config', type=str, default='./config_txt_fewshot.json')
    parser.add_argument('--logDir', type=str, default=logDir)
    parser.add_argument('--resume', type=bool, default=True)
    parser.add_argument('--gpuNum', type=list, default=[0,1])
    parser.add_argument('--batch_size', type=int, default=26) 
    parser.add_argument('--subjects', type=str, default=subjects)
    parser.add_argument('--task', type=str, default='SpokenEEG')
    parser.add_argument('--comments', type=str, default='')
    parser.add_argument('--unseen', type=str, default='stop')
    parser.add_argument('--save_epo', type=int, default=1)

    args = parser.parse_args()
    
    with open(args.config) as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)
    
    random.seed(args.seed)
    args.eval_sub = random.sample(subjects,5)
    
    main(args)        
    
    
    