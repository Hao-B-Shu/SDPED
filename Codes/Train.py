import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torch.optim as optim
from datetime import datetime


def test(dataloader, criterion=None,device='cpu', test_show=None):  # When testing, batch_size should be 1

    global Current_model
    loss=[]
    with torch.no_grad():
        for i, img in enumerate(dataloader):
            pred = Current_model(img['image'].to(device))
            if 'label' in img:
                label=img['label']
                if criterion!=None:
                    label=label.to(device)
                    loss.append(criterion(pred,label).item())
            if test_show!=None and i%test_show==0:
                pred=pred.data.squeeze().cpu()
                plt.imshow(pred,cmap='gray')
                plt.show()
            torch.cuda.empty_cache()
    return loss

def single_train(dataloader, criterion, device='cpu'):
    global Current_model, Optimizer
    loss = []
    torch.cuda.empty_cache()
    for i, data in enumerate(dataloader):
        image = data['image'].to(device)
        label = data['label'].to(device)
        pred = Current_model(image)
        current_loss = criterion(pred, label)
        Optimizer.zero_grad()
        current_loss.backward()
        Optimizer.step()
        loss.append(current_loss.item())

        if i%50==0:
            print(datetime.now(),str(i),np.array(loss).mean())

    return np.array(loss).mean()

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Total', total_num, 'Trainable', trainable_num)

def main(
        train_data_dir: str,
        train_label_dir: str,
        test_data_dir: str,
        test_label_dir: str,
        Pretrain_dir: str,
        criterion: any,
        device: str = 'cpu',
        test_or_not: bool = True,
        epoch: int = 0,
        max_epoch: int = 100,
        refresh: int = 5,
        save_per_epoch: int = 5,
        lr_decrease: any=None,
        train_crop=None,
        train_batch=8,
        train_shuffle=True,
        num_work=8,
        pred_require=16,
        test_show=None,
):

    global Current_model

    if test_or_not == True:

        Current_model.eval()
        Test_data = Generate_Dataset(data_path=test_data_dir, label_path=test_label_dir, test_or_not=test_or_not, crop_size=None,pred_require=pred_require,)
        Test_loader = DataLoader(Test_data, batch_size=1, shuffle=False, num_workers=num_work)
        loss=test(dataloader=Test_loader, device=device,criterion=criterion,test_show=test_show)
        if len(loss)>0:
            print(np.array(loss).mean())

    else:

        Current_model.train()
        get_parameter_number(Current_model)
        Loss_list = []

        torch.manual_seed(100)
        Train_data = Generate_Dataset(data_path=train_data_dir, label_path=train_label_dir, test_or_not=test_or_not,
                                      crop_size=train_crop,pred_require=pred_require)
        Train_loader = DataLoader(Train_data, batch_size=train_batch, shuffle=train_shuffle, num_workers=num_work)

        for Epoch in range(epoch, max_epoch):

            if (lr_decrease!=None and Epoch%lr_decrease==0 and Epoch!=0):
                for params in Optimizer.param_groups:
                    params['lr'] *= 0.1
                    print(params['lr'])

            if Epoch % refresh == 0 and Epoch != 0:
                torch.manual_seed(50 * Epoch + 100)
                Train_data = Generate_Dataset(data_path=train_data_dir, label_path=train_label_dir,
                                              test_or_not=test_or_not, crop_size=train_crop,pred_require=pred_require)
                Train_loader = DataLoader(Train_data, batch_size=train_batch, shuffle=train_shuffle, num_workers=num_work)

            Loss_avg = single_train(dataloader=Train_loader, criterion=criterion, device=device)
            Loss_list.append(Loss_avg)
            if Epoch != epoch and (Epoch - epoch + 1) % save_per_epoch == 0:
                torch.save(Current_model.state_dict(),
                           Pretrain_dir + '/' + 'epoch_' + str(Epoch + 1) + '_loss_' + str(Loss_list[-1]) + '.pth')

            if (Epoch+1) % refresh == 0:
                print('epoch: ' + str(Epoch + 1))
                print('loss=' + str(Loss_list[-1]))

from DataLoad import *  # The standard dataloader
from SDPED_Model import * # Import the model and loss the used function here


########## Set the devices here ##########
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device = 'cuda'
print(device)


###### Set the model here and the optimizer here ####################
Current_model = SDPED(trade_SDB=1,num_block=7)
Optimizer = optim.Adam(filter(lambda p: p.requires_grad, Current_model.parameters()), lr=0.0001, weight_decay=0.00000001)


# Set the checkpoint here if effective
# checkpoint_path=''

# Can use multiple GPU (do not need to delete this even if only one GPU)
Current_model = nn.DataParallel(Current_model)

# Use this to load the checkpoint if effective
# Current_model.load_state_dict(torch.load(checkpoint_path, map_location=device))

# Put the model to the devices
Current_model = Current_model.to(device)


######################### All dirs should be built manually. Codes for saving predictions are included in Others.py. Using test mode here to save is not recommended###########################
main(
    train_data_dir='',# Set the path of the training data here (invalid if test_or_not=True)
    train_label_dir='',# Set the path of the training labels here (invalid if test_or_not=True)
    test_data_dir='',# Set the path of the test data here (invalid if test_or_not=False)
    test_label_dir='',# Set the path of the test labels here (invalid if test_or_not=False, and can be None)
    Pretrain_dir='',# Set the path of the dir for saving checkpoints
    device=device,
    test_or_not=False,# test_or_not=True for testing, test_or_not=False for training
    test_show=None, # Show prediction per test_show, e. g. test_show=50 represents show predictions per 50 image in test mode (invalid if test_or_not=False)
    epoch=0, # The beginning epoch in training
    max_epoch=100,# The ending epoch in training
    refresh=5,# Data will be refreshed every refresh in training, e. g. refresh=5 represents training data will be reloaded every 5 epoches
    save_per_epoch=5,# Checkpoints will be saved every save_per_epoch in training, e. g. save_per_epoch=5 represents checkpoints will be saved every 5 epoches
    lr_decrease=50, # After lr_decrease, learn rate will be devided by 10. e. g. lr_decrease=50 represents learn rate will be decreased every 50 epoches
    criterion=WBCE, # Set the loss function here
    train_crop=[320,320], # Set the size of image in training here, training images will be randomly cropped to the size in training
    train_batch=8,# Set the number of training batches here
    train_shuffle=True,
    num_work=8,
    pred_require=16 # Set the required size for input image here (only valid in prediction, and for training, please set the size in train_crop), the images (both height and width) will be resized to a multiple of it. e. g. if the model requires the size of input image to be a multiple of 16, then set pred_require=16 here.
)
torch.cuda.empty_cache()
###########################################
