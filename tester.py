""" code for testing the model """



import time
import torch
from torchvision.utils import make_grid 
from tqdm import tqdm

from utils import miou
from torch.utils.tensorboard import SummaryWriter

class tester:
    def __init__(self, model, dataloader, criterion, optimizer,run_id, log_freq, epochs=1 ):

        self.model = model #loaded torch model 
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs

        self.run_id = run_id
        self.log_freq = log_freq


    def test(self):
        """ This function is used to test a model
        
        """

        self.model.eval()
        since = time.time()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model.to(device)
        
        writer = SummaryWriter(log_dir = f"./runs/{self.run_id}")
        global_step = 0

        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')

            self.model.eval()

            running_loss = 0
            running_iou = 0

            #iterating over data 
            for idx, sample in enumerate(tqdm(self.dataloader)):
                voxel = sample['voxel'].to(device)
                global_step += 1
                voxel = voxel.unsqueeze(1)
                gt = sample['gt'].to(device) #torch.Size([16, 128, 128, 8])
                

                self.optimizer.zero_grad()

                
                with torch.set_grad_enabled(False):
                    #forward call 
                    output = self.model(voxel)
                    loss = self.criterion(output.squeeze(), gt)
                    output = torch.sigmoid(output)
                    
                
                #stats
                running_loss += loss.item() 
                output_miou = output.squeeze().detach()
                output_miou[output_miou >= 0.5] = 1
                output_miou[output_miou < 0.5] = 0

                batch_miou  = miou(output_miou, gt).item()
                running_iou += batch_miou

                #logging to tensorboard 
                writer.add_scalar("Losses/test_running_loss", loss.item(), global_step)
                writer.add_scalar("Losses/test_running_iou", batch_miou, global_step)

                if True:  
                # if idx == len(self.dataloader)-1:  
                # if global_step  % self.log_freq == 0:
                    test_image = make_grid(voxel[:16,...,0], nrow = 4)
                    pred_image = make_grid(output[:16,...,0], nrow = 4)
                    gt_image = make_grid(gt[:16,None, ...,  0], nrow = 4)

                    # breakpoint()
                    writer.add_image("Test_Images/test_image", test_image, global_step)
                    writer.add_image("Test_Images/pred_image", pred_image, global_step)
                    writer.add_image("Test_Images/gt_image", gt_image, global_step)



            epoch_loss = running_loss / len(self.dataloader)
            epoch_mean_iou = running_iou / len(self.dataloader)
            
            #logging to tensorboard 
            writer.add_scalar("Losses/test_epoch_loss", epoch_loss, epoch)
            writer.add_scalar("Losses/test_epoch_mean_iou", epoch_mean_iou, epoch)
            print('{} Loss: {:.4f} IoU: {:.4f}'.format('test', epoch_loss, epoch_mean_iou))


        time_elapsed = time.time() - since
        print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

