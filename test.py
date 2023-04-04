from dataloader import ElectronMicroscopyDataset_Test
from model import UNet3D
from trainer import trainer
from tester import tester

import torch
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_img_tif", type=str, default=f"./data/testing.tif")
    parser.add_argument("--test_gt_tif", type=str, default=f"./data/testing_groundtruth.tif")

    parser.add_argument("--crop_size", type=str, default='[128,128]')
    parser.add_argument("--slices", type=int, default=8) 
    parser.add_argument("--batch_size", type=int, default=16) 
    parser.add_argument("--learning_rate", type=float, default=3e-4) 

    parser.add_argument("--NUM_EPOCHS", type=int, default=10) 
    parser.add_argument("--RUN_ID", type=str, default='debug_test') 
    parser.add_argument("--LOG_FREQ", type=int, default=10) 
    parser.add_argument("--mode", type=str, default='test', choices = ['val', 'test']) 


    args = parser.parse_args()

    MODEL_PATH = f'./checkpoints/{args.RUN_ID}.pth' #TODO
    IN_CHANNELS = 1
    NUM_CLASSES = 1

    # #create dataloaders
    test_dataset =  ElectronMicroscopyDataset_Test( args.test_img_tif, 
                                                    args.test_gt_tif, 
                                                    eval(args.crop_size), 
                                                    args.slices, 
                                                    args.batch_size,
                                                    args.mode)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle = False)

    #load model 
    model = torch.load(MODEL_PATH)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    #create tester and running inference 
    tester = tester(    model=model, 
                        dataloader=test_dataloader, 
                        criterion=criterion, 
                        optimizer= optimizer, 
                        run_id = args.RUN_ID,
                        log_freq = args.LOG_FREQ)
    tester.test()