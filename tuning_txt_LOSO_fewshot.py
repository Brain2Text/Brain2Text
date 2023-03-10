import os
import torch
from models import models_txt as networks
from modules import GreedyCTCDecoder, AttrDict, RMSELoss, save_checkpoint
from utils import  word_index
import torch.nn as nn
from sklearn.model_selection import train_test_split
from NeuroTalkDataset import myDataset
import time
import torch.optim.lr_scheduler
import numpy as np
import torchaudio
from torchmetrics import CharErrorRate
import json
import argparse
import random
from torch.utils.tensorboard import SummaryWriter
from train_txt_fewshot import train_G, train_D


def train(args, train_loader, models, criterions, optimizers, epoch, trainValid=True):
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
    
    # Get only few-shot data    
    input, target_cl, voice = next(iter(train_loader))


    input = input.cuda()
    target_cl = target_cl.cuda()
    voice = voice.cuda()
    labels = torch.argmax(target_cl,dim=1) 
    
    # extract few-shot
    idx_shot=[]
    dump = list(range(len(input)))
    _,idx_shot,_,_ = train_test_split(dump,labels.cpu(),test_size=args.fewshot*13,shuffle=True, stratify=labels.cpu(), random_state=args.seed)

    input = input[idx_shot]
    target_cl = target_cl[idx_shot]
    voice = voice[idx_shot]
    labels = labels[idx_shot]

    # unseen extract
    idx_unseen,idx_seen = [], []
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
    
    # Seen
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

    epoch_loss_g_ns = np.array(epoch_loss_g_ns)
    epoch_acc_g_ns = np.array(epoch_acc_g_ns)
    epoch_loss_d_ns = np.array(epoch_loss_d_ns)
    epoch_acc_d_ns = np.array(epoch_acc_d_ns)

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
    if trainValid:
        tag = 'tune-train'
    else:
        tag = 'tune-valid'
        
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



def eval(args, train_loader, models, criterions, optimizers, epoch, trainValid=True):
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
    
    for i, (input, target_cl, voice) in enumerate(train_loader):    

        print("\rBatch [%5d / %5d]"%(i+1,total_batches), sep=' ', end='', flush=True)
        
        input = input.cuda()
        target_cl = target_cl.cuda()
        voice = torch.squeeze(voice,dim=-1).cuda()
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
                                                                 epoch,
                                                                 False)
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
    if trainValid:
        tag = 'tune-train'
    else:
        tag = 'tune-valid'
        
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



def save_test_all(args, test_loader, models, save_idx=None):
    
    model_g = models[0].eval()
    decoder_STT = models[3]
    
    save_idx=0
    for i, (input, target_cl, voice) in enumerate(test_loader):
        
        # subject append
        input = torch.reshape(input, (input.shape[0]*input.shape[1],input.shape[2],input.shape[3]))
        target_cl = torch.reshape(target_cl, (target_cl.shape[0]*target_cl.shape[1],target_cl.shape[2]))
    
        input = input.cuda()
        labels = torch.argmax(target_cl,dim=1)    
        
        with torch.no_grad():
            # run the mdoel
            em_recon = model_g(input)
        
        # decoder STT
        transcript_recon = []
        for batch_idx in range(len(input)):
            transcript = decoder_STT(em_recon[batch_idx])
            transcript_recon.append(transcript)
            
            str_tar = args.word_label[labels[batch_idx].item()].replace("|", ",")
            str_tar = str_tar.replace(" ", ",")
            
            str_pred = transcript_recon[batch_idx].replace("|", ",")
            str_pred = str_pred.replace(" ", ",")
            
            # word save 
            args.logger_eval.write("\n%d\t%s\t%s"
                              %(save_idx, str_tar, str_pred ))
            args.logger_eval.flush()
            
            save_idx=save_idx+1


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

    # optimizer
    optimizer_g = torch.optim.AdamW(model_g.parameters(), lr=args.lr_g_tune, betas=(0.8, 0.99), weight_decay=0.01)
    optimizer_d = torch.optim.AdamW(model_d.parameters(), lr=args.lr_d_tune, betas=(0.8, 0.99), weight_decay=0.01)
   
    scheduler_g = torch.optim.lr_scheduler.CyclicLR(optimizer_g, base_lr=args.lr_g_tune/2, max_lr=args.lr_g_tune*2, 
                                                    step_size_up=10, step_size_down=None, 
                                                    mode='triangular2', gamma=args.lr_g_tune_decay, 
                                                    scale_fn=None, scale_mode='cycle', cycle_momentum=False, 
                                                    base_momentum=0.8, max_momentum=0.9, last_epoch=- 1, verbose=False)
    scheduler_d = torch.optim.lr_scheduler.CyclicLR(optimizer_d, base_lr=args.lr_d_tune/2, max_lr=args.lr_d_tune*2, 
                                                    step_size_up=10, step_size_down=None, 
                                                    mode='triangular2', gamma=args.lr_d_tune_decay, 
                                                    scale_fn=None, scale_mode='cycle', cycle_momentum=False, 
                                                    base_momentum=0.8, max_momentum=0.9, last_epoch=- 1, verbose=False)

    # Directory
    logDir = args.logDir + str(args.fewshot) + '-shot/'
    saveDir = logDir + 'LOSO_fineTune_sub' + str(args.sub) + '_' + args.task + args.comments
    args.savemodel = args.pretrain_model + '/savemodel'
    args.tunemodel = saveDir + '/tunemodel'
    
    
    # Load trained model
    start_epoch = 0
    start_best = 1000
    
    loc_g = os.path.join(args.savemodel, 'BEST_checkpoint_g.pt')
    loc_d = os.path.join(args.savemodel, 'BEST_checkpoint_d.pt')
    
    if os.path.isfile(loc_g):
        print("=> loading checkpoint '{}'".format(loc_g))
        checkpoint_g = torch.load(loc_g, map_location='cpu')
        model_g.load_state_dict(checkpoint_g['state_dict'])
        print('Load {}th epoch model'.format(checkpoint_g['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(loc_g))
        return 0

    if os.path.isfile(loc_d):
        print("=> loading checkpoint '{}'".format(loc_d))
        checkpoint_d = torch.load(loc_d, map_location='cpu')
        model_d.load_state_dict(checkpoint_d['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(loc_d))
        return 0
        
    if args.resume:
        loc_g = os.path.join(args.tunemodel, 'checkpoint_g.pt')
        loc_d = os.path.join(args.tunemodel, 'checkpoint_d.pt')

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

    # create the directory if not exist
    if not os.path.exists(logDir):
        os.mkdir(logDir)
        
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)
        
    if not os.path.exists(args.tunemodel):
        os.mkdir(args.tunemodel)
        
    args.saveword = saveDir + '/epoword'
    if not os.path.exists(args.saveword):
        os.mkdir(args.saveword)
    
    args.logs = saveDir + '/logs'
    if not os.path.exists(args.logs):
        os.mkdir(args.logs)

    args.args = saveDir + '/args'
    if not os.path.exists(args.args):
        os.mkdir(args.args)
        
    # log save
    logFileLoc = args.saveword + '/tune_pred_words.txt'
    if os.path.isfile(logFileLoc):
        args.logger = open(logFileLoc, 'a')
    else:
        args.logger = open(logFileLoc, 'w')
        args.logger.write("%s\t%s\t%s"
                          %('idx', 'gt', 'pred'))
    args.logger.flush()
    
    # log save
    logFileLoc = args.saveword + '/eval_pred_words.txt'
    if os.path.isfile(logFileLoc):
        args.logger_eval = open(logFileLoc, 'a')
    else:
        args.logger_eval = open(logFileLoc, 'w')
        args.logger_eval.write("%s\t%s\t%s"
                          %('idx', 'gt', 'pred'))
    args.logger_eval.flush()
    
    # arguments save
    argsFileLoc = args.args + '/arguments.txt'
    argsave = open(argsFileLoc, 'a')
    argsave.write("------Tuning------\nsub: %02d \ntask: %s \nmax_epochs: %s \
                  \nl_g: %s \nlr_g: %s \nlr_g_decay: %s\nlr_g_tune: %s\nlr_g_tune_decay: %s\
                      \nl_d: %s \nlr_g: %s \nlr_d_decay: %s\nlr_d_tune: %s\nlr_d_tune_decay: %s\
                          \nbatch size: %s \nunseen word: %s\
                              \ndata Loc: %s \nlog Loc: %s"
                      %(args.sub, args.task, args.max_epochs, 
                        args.l_g, args.lr_g, args.lr_g_decay, args.lr_g_tune, args.lr_g_tune_decay, 
                        args.l_d, args.lr_d, args.lr_d_decay, args.lr_d_tune, args.lr_d_tune_decay, 
                        args.batch_size, args.unseen,
                        args.dataLoc, logDir))
    argsave.flush()
    
     
    # Tensorboard setting
    args.writer = SummaryWriter(args.logs)
    
    # fewshot?
    if args.fewshot == -1:
        args.batch_tune = args.batch_size
    else:
        args.batch_tune = 780
            
    # Data loader define
    generator = torch.Generator().manual_seed(args.seed)

    trainset = myDataset(mode=0, data=args.dataLoc+'/sub%02d'%args.sub, task=args.task)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_tune, shuffle=True, generator=generator, num_workers=4*len(args.gpuNum), pin_memory=True)
    
    valset = myDataset(mode=2, data=args.dataLoc+'/sub%02d'%args.sub, task=args.task)
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
        
        
        print('\nTrain')
        if args.fewshot == -1:
            Tr_losses = eval(args, train_loader, 
                               (model_g, model_d, model_STT, decoder_STT), 
                               (criterion_recon, criterion_ctc, criterion_adv, criterion_cl, RMSE, CER), 
                               (optimizer_g, optimizer_d), 
                               epoch,
                               True)
        else:
            Tr_losses = train(args, train_loader, 
                               (model_g, model_d, model_STT, decoder_STT), 
                               (criterion_recon, criterion_ctc, criterion_adv, criterion_cl, RMSE, CER), 
                               (optimizer_g, optimizer_d), 
                               epoch,
                               True)
        
        print('\nValid')
        Val_losses = eval(args, val_loader, 
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
        
        save_checkpoint(state_g, is_best, args.tunemodel, 'checkpoint_g.pt')
        save_checkpoint(state_d, is_best, args.tunemodel, 'checkpoint_d.pt')

        time_taken = time.time() - start_time
        print("Time: %.2f\n"%time_taken)

    args.writer.flush()
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
    parser.add_argument('--gpuNum', type=list, default=[0,1,2])
    parser.add_argument('--batch_size', type=int, default=780) 
    parser.add_argument('--fewshot', type=int, default=-1) # All: -1
    parser.add_argument('--subjects', type=str, default=subjects)
    parser.add_argument('--subNum', type=int, default=subNum)
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
    args.subjects = random.sample(subjects,5)
    args.sub = args.subjects[args.subNum]
    
    main(args)        
    
    
    