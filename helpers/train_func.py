import sys
sys.path.append("/workspace/Documents")
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from ema_pytorch import EMA
from accelerate import Accelerator
import random
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
import HFpEF_CMR_GraphStrain.functions_collection as ff


class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight, logits = False):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.logits = logits

    def forward(self, y_pred, y_true):
        return F.binary_cross_entropy(y_pred, y_true, weight=self.pos_weight.to(y_pred.device))
     


class Trainer(object):
    def __init__(
        self,
        model,
        loss,
        generator_train, 
        generator_val,
        train_batch_size = 5,

        train_num_steps= 1000,
        save_folder = '',
        train_lr_decay_every = 1000,
        save_models_every = 2,
        lr = 1e-4,
        weight_decay = 1e-5,):
    
        super().__init__()
        self.accelerator = Accelerator(
            split_batches = True,
            mixed_precision = 'no',)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#;print('device:', self.device)

        self.model = model
        self.strain = model.strain # important
        # print('in trainer, strain:', self.strain)
        self.loss = loss
        self.save_folder = save_folder; os.makedirs(self.save_folder, exist_ok=True)

        # dataloader
        self.generator_train = generator_train
        dl = DataLoader(self.generator_train, batch_size = train_batch_size, shuffle = False, pin_memory = True, num_workers = 0)
        self.dl = self.accelerator.prepare(dl)
        self.EHR = generator_train.EHR 

        self.generator_val = generator_val
        dl_val = DataLoader(self.generator_val, batch_size = train_batch_size, shuffle = False, pin_memory = True, num_workers = 0)# cpu_count())
        self.dl_val = self.accelerator.prepare(dl_val)

        # optimizer
        self.opt = Adam(self.model.parameters(), lr = lr, betas = (0.9, 0.99), weight_decay = weight_decay)
        self.scheduler = StepLR(self.opt, step_size = 1, gamma=0.95)
        self.max_grad_norm = 1.
        self.train_num_steps = train_num_steps
        self.train_lr_decay_every = train_lr_decay_every
        self.save_model_every = save_models_every
        self.step = 0

        if self.accelerator.is_main_process:
            self.ema = EMA(self.model, beta = 0.995, update_every = 10)
            self.ema.to(self.device)
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)


    def save(self, stepNum):
        if not self.accelerator.is_local_main_process:
            return
        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'decay_steps': self.scheduler.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if ff.exists(self.accelerator.scaler) else None,}
        
        torch.save(data, os.path.join(self.save_folder, 'model-' + str(stepNum) + '.pt'))

    def load_model(self, trained_model_filename):
        accelerator = self.accelerator
        device = accelerator.device
        data = torch.load(trained_model_filename, map_location=device)
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        self.scheduler.load_state_dict(data['decay_steps'])
        if ff.exists(self.accelerator.scaler) and ff.exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])


    def train(self, pre_trained_model = None ,start_step = None):
        accelerator = self.accelerator
        device = accelerator.device

        # load pre-trained
        if pre_trained_model is not None:
            self.load_model(pre_trained_model)
            print('model loaded from ', pre_trained_model)

        if start_step is not None:
            self.step = start_step

        self.scheduler.step_size = 1
        val_loss = np.inf; accuracy = np.inf; sensitivity = np.inf; specificity = np.inf; accuracy_val = np.inf; sensitivity_val = np.inf; specificity_val = np.inf
        training_log = []

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:
            
            while self.step < self.train_num_steps:
                average_loss = []
                count = 0

                for batch in self.dl:
                    self.opt.zero_grad()
                    batch_ecc, batch_err,batch_err_padded, batch_y, batch_ehr = batch
                    data_ecc, data_err, data_err_padded, data_y, data_ehr = batch_ecc.to(device), batch_err.to(device), batch_err_padded.to(device), batch_y.to(device), batch_ehr.to(device)
                   
                    with self.accelerator.autocast():
                        if self.strain == 'Ecc' or self.strain == 'Err':
                            output = self.model(data_ecc) if self.strain == 'Ecc' else self.model(data_err)
                        elif self.strain == 'both':
                            if self.EHR is None:
                                output = self.model(data_ecc, data_err)
                            else:
                                output = self.model(data_ecc, data_err, data_ehr)
                        # calculate loss
                        loss = self.loss(output, data_y)
                        
                    average_loss.append(loss.item())
                   
                    self.accelerator.backward(loss)
                    self.opt.step() 

                average_loss = sum(average_loss) / len(average_loss)
                pbar.set_description(f'average loss: {average_loss:.4f}')

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.step += 1

                # save the model
                if self.step !=0 and self.step % self.save_model_every == 0:
                   self.save(self.step)
                # update the parameter
                if self.step !=0 and self.step % self.train_lr_decay_every ==0:
                    self.scheduler.step()

                self.ema.update()

                # do the validation if necessary
                if self.step !=0 and self.step % self.save_model_every == 0:
                    self.model.eval()
                    with torch.no_grad():
                        val_loss = []
                        for batch in self.dl_val:
                            batch_ecc, batch_err,batch_err_padded, batch_y, batch_ehr = batch
                            data_ecc, data_err, data_err_padded, data_y, data_ehr = batch_ecc.to(device), batch_err.to(device), batch_err_padded.to(device), batch_y.to(device), batch_ehr.to(device)
                            with self.accelerator.autocast():
                                if self.strain == 'Ecc' or self.strain == 'Err':
                                    output = self.model(data_ecc) if self.strain == 'Ecc' else self.model(data_err)
                                elif self.strain == 'both':
                                    if self.EHR is None:
                                        output = self.model(data_ecc, data_err)
                                    else:
                                        output = self.model(data_ecc, data_err, data_ehr)
                                loss = self.loss(output, data_y)
                            val_loss.append(loss.item())
                        val_loss = sum(val_loss) / len(val_loss)
                
                
                    dl_train_in_calc = DataLoader(self.generator_train, batch_size = 1, shuffle = False, pin_memory = True, num_workers = 0)
                    y_pred_list = []; y_pred_proba_list = []; y_true_list = []
                    for batch in dl_train_in_calc:
                        batch_ecc, batch_err,batch_err_padded, batch_y, batch_ehr = batch
                        data_ecc, data_err, data_err_padded, data_y, data_ehr = batch_ecc.to(device), batch_err.to(device), batch_err_padded.to(device), batch_y.to(device), batch_ehr.to(device)
   
                        if self.strain == 'Ecc' or self.strain == 'Err':
                            y_pred_prob = self.model(data_ecc) if self.strain == 'Ecc' else self.model(data_err)
                        elif self.strain == 'both':
                            if self.EHR is None:
                                y_pred_prob = self.model(data_ecc, data_err)
                            else:
                                y_pred_prob = self.model(data_ecc, data_err, data_ehr)
        
                        y_pred_proba_list.append(y_pred_prob.item())
                        y_pred = 1 if y_pred_prob.item() > 0.5 else 0
                        y_pred_list.append(y_pred)
                        y_true_list.append(data_y.item())
                    # calculate accuracy using predict_collect and ground truth
                    accuracy, sensitivity, specificity,_,_,_,_ = ff.quantitative(np.asarray(y_pred_list), np.asarray(y_true_list))

                    # do for val
                    dl_val_in_calc = DataLoader(self.generator_val, batch_size = 1, shuffle = False, pin_memory = True, num_workers = 0)
                    y_pred_list_val = []; y_pred_proba_list_val = []; y_true_list_val = []
                    for batch in dl_val_in_calc:
                        batch_ecc, batch_err,batch_err_padded, batch_y, batch_ehr = batch
                        data_ecc, data_err, data_err_padded, data_y, data_ehr = batch_ecc.to(device), batch_err.to(device), batch_err_padded.to(device), batch_y.to(device), batch_ehr.to(device)
                        if self.strain == 'Ecc' or self.strain == 'Err':
                            y_pred_prob = self.model(data_ecc) if self.strain == 'Ecc' else self.model(data_err)
                        elif self.strain == 'both':
                            if self.EHR is None:
                                y_pred_prob = self.model(data_ecc, data_err)
                            else:
                                y_pred_prob = self.model(data_ecc, data_err, data_ehr)
                        y_pred_proba_list_val.append(y_pred_prob.item())
                        y_pred = 1 if y_pred_prob.item() > 0.5 else 0
                        y_pred_list_val.append(y_pred)
                        y_true_list_val.append(data_y.item())
                    accuracy_val, sensitivity_val, specificity_val,_,_,_,_ = ff.quantitative(np.asarray(y_pred_list_val), np.asarray(y_true_list_val))
                    # print('train accuracy:', accuracy,' sensitivity:', sensitivity, ' specificity:', specificity)
                    # print('val accuracy:', accuracy_val, ' sensitivity:', sensitivity_val, ' specificity:', specificity_val)
                    # print('train loss:', average_loss, ' val loss:', val_loss)
                    
                    # save the training log
                    training_log.append([self.step,self.scheduler.get_last_lr()[0],average_loss,val_loss,accuracy,sensitivity,specificity,accuracy_val,sensitivity_val,specificity_val])
                    df = pd.DataFrame(training_log,columns = ['iteration','learning_rate','train_loss','val_loss','train_accuracy','train_sensitivity','train_specificity','val_accuracy','val_sensitivity','val_specificity'])
                    log_folder = os.path.join(os.path.dirname(self.save_folder),'log');ff.make_folder([log_folder])
                    df.to_excel(os.path.join(log_folder, 'training_log.xlsx'),index=False)
                    
                    self.model.train(True)

                # at the end of each epoch, call on_epoch_end
                self.generator_train.on_epoch_end(); self.generator_val.on_epoch_end()
                pbar.update(1)
        accelerator.print('training complete')

