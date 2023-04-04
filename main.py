from dataloader import ElectronMicroscopyDataset, ElectronMicroscopyDataset_Test
from model import UNet3D
from trainer import trainer
from tester import tester

import torch
import argparse


if __name__ == '__main__':
    #hyperparameters     
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_img_tif", type=str, default=f"./data/training.tif")
    parser.add_argument("--train_gt_tif", type=str, default=f"./data/training_groundtruth.tif")


    parser.add_argument("--crop_size", type=str, default='[128,128]')
    parser.add_argument("--slices", type=int, default=8) 
    parser.add_argument("--batch_size", type=int, default=16) 
    parser.add_argument("--learning_rate", type=float, default=3e-4) 

    parser.add_argument("--NUM_EPOCHS", type=int, default=10) 
    parser.add_argument("--RUN_ID", type=str, default='debug') 
    parser.add_argument("--LOG_FREQ", type=int, default=10) 
    parser.add_argument("--mode", type=str, default='val', choices = ['val', 'test']) 

    args = parser.parse_args()

    MODEL_PATH = f'./checkpoints/{args.RUN_ID}.pth' 
    IN_CHANNELS = 1 
    NUM_CLASSES = 1 

    #create dataloaders
    train_dataset =  ElectronMicroscopyDataset(args.train_img_tif, args.train_gt_tif, eval(args.crop_size), args.slices, args.batch_size)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle = True)
    
    val_dataset =  ElectronMicroscopyDataset_Test(  args.train_img_tif, 
                                                    args.train_gt_tif, 
                                                    eval(args.crop_size), 
                                                    args.slices, 
                                                    args.batch_size, 
                                                    args.mode)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle = True)

    
    #create model 
    model = UNet3D(in_channels=IN_CHANNELS , num_classes= NUM_CLASSES)

    #training the model 
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    trainer = trainer(  model=model, 
                        train_dataloader=train_dataloader, 
                        val_dataloader = val_dataloader,
                        criterion=criterion, 
                        optimizer= optimizer, 
                        epochs= args.NUM_EPOCHS,
                        run_id = args.RUN_ID,
                        log_freq = args.LOG_FREQ)

    trainer.train()

    #saving the model 
    model_path = MODEL_PATH 
    torch.save(model, model_path)

    

    