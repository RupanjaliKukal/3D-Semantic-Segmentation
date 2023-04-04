""" class for training the model """

import time
import torch
from torchvision.utils import make_grid 
from tqdm import tqdm

from utils import miou
from torch.utils.tensorboard import SummaryWriter

class trainer:
    def __init__(self, model, train_dataloader, val_dataloader,  criterion, optimizer, epochs, run_id, log_freq):

        self.model = model 
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs

        self.run_id = run_id
        self.log_freq = log_freq


    def train(self):
        """ This function is used to train a model
        Returns:
            None 
        """

        since = time.time()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model.to(device)
        
        writer = SummaryWriter(log_dir = f"./runs/{self.run_id}")
        global_step_train = 0
        global_step_val = 0

        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')

            self.model.train()

            running_loss = 0
            running_iou = 0

            #iterating over data 
            for idx, sample in enumerate(tqdm(self.train_dataloader)):
                voxel = sample['voxel'].to(device)
                global_step_train += 1
                voxel = voxel.unsqueeze(1)
                gt = sample['gt'].to(device) #torch.Size([16, 128, 128, 8])
                

                self.optimizer.zero_grad()

                
                with torch.set_grad_enabled(True):
                    #forward call 
                    output = self.model(voxel)
                    loss = self.criterion(output.squeeze(), gt)
                    output = torch.sigmoid(output)
                    
                    #backward call
                    loss.backward()
                    self.optimizer.step()
                
                #stats
                running_loss += loss.item() 
                output_miou = output.squeeze().detach()
                output_miou[output_miou > 0.5] = 1
                output_miou[output_miou < 0.5] = 0

                batch_miou  = miou(output_miou, gt).item()
                running_iou += batch_miou

                #logging to tensorboard 
                writer.add_scalar("Train_Losses/train_running_loss", loss.item(), global_step_train)
                writer.add_scalar("Train_Losses/train_running_iou", batch_miou, global_step_train)

                
                if global_step_train  % self.log_freq == 0:
                    train_image = make_grid(voxel[:16,...,0], nrow = 4)
                    pred_image = make_grid(output[:16,...,0], nrow = 4)
                    gt_image = make_grid(gt[:16,None, ...,  0], nrow = 4)

                    writer.add_image("Train_Images/train_image", train_image, global_step_train)
                    writer.add_image("Train_Images/pred_image", pred_image, global_step_train)
                    writer.add_image("Train_Images/gt_image", gt_image, global_step_train)


            epoch_loss = running_loss / len(self.train_dataloader)
            epoch_mean_iou = running_iou / len(self.train_dataloader)
            
            #logging to tensorboard 
            writer.add_scalar("Train_Losses/train_epoch_loss", epoch_loss, epoch)
            writer.add_scalar("Train_Losses/train_epoch_mean_iou", epoch_mean_iou, epoch)

            print('{} Loss: {:.4f} IoU: {:.4f}'.format('Train', epoch_loss, epoch_mean_iou))

            #saving checkpoint 
            model_path = f"./checkpoints/{self.run_id}_{global_step_train}.pth"
            torch.save(self.model, model_path)

#______________________________________________________________________
            #VALIDATION LOOP
            self.model.eval()

            running_loss_val = 0
            running_iou_val = 0

            #iterating over data 
            for idx, sample in enumerate(tqdm(self.val_dataloader)):
                voxel = sample['voxel'].to(device)
                global_step_val += 1
                voxel = voxel.unsqueeze(1)
                gt = sample['gt'].to(device) #torch.Size([16, 128, 128, 8])

                self.optimizer.zero_grad()

                with torch.set_grad_enabled(False):
                    #forward call 
                    output = self.model(voxel)
                    loss = self.criterion(output.squeeze(), gt)
                    output = torch.sigmoid(output)
                    
                
                #stats
                running_loss_val += loss.item() 
                output_miou = output.squeeze().detach()
                output_miou[output_miou >= 0.5] = 1
                output_miou[output_miou < 0.5] = 0

                batch_miou  = miou(output_miou, gt).item()
                print(f"Validation {idx} miou: {batch_miou}")
                running_iou_val += batch_miou

                #logging to tensorboard 
                writer.add_scalar("Val_Losses/test_running_loss", loss.item(), global_step_val)
                writer.add_scalar("Val_Losses/test_running_iou", batch_miou, global_step_val)


                if True:
                    test_image = make_grid(voxel[:16,...,0], nrow = 4)
                    pred_image = make_grid(output[:16,...,0], nrow = 4)
                    gt_image = make_grid(gt[:16, None,...,  0], nrow = 4)

                    writer.add_image("Val_Images/test_image", test_image, global_step_val)
                    writer.add_image("Val_Images/pred_image", pred_image, global_step_val)
                    writer.add_image("Val_Images/gt_image", gt_image, global_step_val)


            epoch_loss_val = running_loss_val / len(self.val_dataloader)
            epoch_mean_iou_val = running_iou_val / len(self.val_dataloader)
            
            #logging to tensorboard 
            writer.add_scalar("Val_Losses/test_epoch_loss", epoch_loss_val, epoch)
            writer.add_scalar("Val_Losses/test_epoch_mean_iou", epoch_mean_iou_val, epoch)
            # breakpoint()

            print('{} Loss: {:.4f} IoU: {:.4f}'.format('Val', epoch_loss_val, epoch_mean_iou_val))



        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

