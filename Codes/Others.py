import os
import cv2
import numpy as np
import torch
from torchvision import transforms as T
from scipy import io

normalize=T.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
totensor=T.ToTensor()


######################################Save###################################
def Save_Predict(model, Data_dir, Save_pred_dir,device='cpu',pred_require=16):

    model.to(device)
    model.eval()

    Data_name = os.listdir(Data_dir)
    Data_name.sort()

    for index in range(len(Data_name)):
        Data_file = os.path.join(Data_dir, Data_name[index])
        Data = cv2.imread(Data_file, cv2.IMREAD_COLOR)
        Data = cv2.cvtColor(Data, cv2.COLOR_BGR2RGB)

        h,w,c=Data.shape

        h_r=h//pred_require*pred_require
        w_r=w//pred_require*pred_require

        if h!=h_r or w!=w_r:
            Data=cv2.resize(Data,(w_r,h_r))

        Data=T.ToTensor()(Data)
        Data=normalize(Data).unsqueeze(0).to(device)
        Out = model(Data)
        if type(Out)==type([]):
            Out=Out[-1]
        Out = Out.squeeze().data.cpu().numpy()

        if h != h_r or w != w_r:
            Out = cv2.resize(Out, (w, h)).astype(np.float32)

        Out = Unit_Normalize(Out)
        cv2.imwrite(os.path.join(Save_pred_dir, Data_name[index][:-4] + '.png'), Out)

        torch.cuda.empty_cache()

def Save_label_to_mat(Label_dir,Save_dir):#Nonzero value will set be 1
    Label_name=os.listdir(Label_dir)
    Label_name.sort()
    for index in range (len(Label_name)):
        Label_file=os.path.join(Label_dir, Label_name[index])
        Label =cv2.imread(Label_file, cv2.IMREAD_GRAYSCALE)
        Label[Label!=0]=1
        io.savemat(Save_dir+'/'+Label_name[index][:-4]+'.mat',{'groundTruth':[{'Boundaries':Label}]})

def Add_label_to_data(Label_dir,Data_dir):
    Label_name=os.listdir(Label_dir)
    Label_name.sort()
    for index in range (len(Label_name)):
        Label_file=os.path.join(Label_dir, Label_name[index])
        img=cv2.imread(Label_file)
        Label =cv2.imread(Label_file, cv2.IMREAD_GRAYSCALE)
        Label[Label!=0]=255
        img[img != 0] = 255
        cv2.imwrite(os.path.join(Label_dir, 'Label'+Label_name[index][:-4] + '.png'), Label.astype(np.uint8))
        cv2.imwrite(os.path.join(Data_dir, 'Label' + Label_name[index][:-4] + '.jpg'), img.astype(np.uint8))


##################################Preprocess################################################
def Check_name(Data_dir,Edge_dir):
    Data_name = os.listdir(Data_dir)
    Edge_name = os.listdir(Edge_dir)
    Data_name.sort()
    Edge_name.sort()
    assert len(Data_name)==len(Edge_name)
    for index in range (len(Data_name)):
        assert Data_name[index][0:-4]==Edge_name[index][0:-4]
    print('Check passed')
    print('Number',len(Data_name))

def Unit_Normalize(img,eps=0.0000000000000001):
    a=torch.zeros(1,1)
    b=np.zeros((1,1))
    if type(img)==type(b):
        return (((img - np.min(img)) / (np.max(img) - np.min(img)+eps)) * 255).astype(np.uint8)
    elif type(img)==type(a):
        if len(img.shape)==2 or len(img.shape)==3:
            return ((img - torch.min(img))  / (torch.max(img) - torch.min(img)+eps))* 255
        elif len(img.shape)==4:
            B,C,H,W=img.shape
            IMG=[]
            for i in range (B):
                IMG.append(((img[i] - torch.min(img[i]))  / (torch.max(img[i]) - torch.min(img[i])+eps))* 255)
            return torch.stack(IMG,dim=0)

def Half_w(image,label):
    H,W,_=image.shape
    new_w=int(W/2)
    Img1=image[:,0:new_w,:]
    Img2 = image[:, new_w:2*new_w, :]
    Lab1 = label[:, 0:new_w]
    Lab2 = label[:, new_w:2 * new_w]
    return Img1,Img2,Lab1,Lab2

def Half_h(image,label):
    H,W,_=image.shape
    new_h=int(H/2)
    Img1=image[0:new_h,:,:]
    Img2 = image[ new_h:2*new_h,:, :]
    Lab1 = label[0:new_h,: ]
    Lab2 = label[ new_h:2 * new_h,:]
    return Img1,Img2,Lab1,Lab2

def Data_aug_half(img_dir,lab_dir,imgaug_dir,labaug_dir,direction='auto',thr=320):
    img_name = sorted(os.listdir(img_dir))
    lab_name = sorted(os.listdir(lab_dir))

    for index in range (len(img_name)):
        Image_file = os.path.join(img_dir, img_name[index])
        Label_file = os.path.join(lab_dir, lab_name[index])
        img = cv2.imread(Image_file, cv2.IMREAD_COLOR)
        lab = cv2.imread(Label_file, cv2.IMREAD_GRAYSCALE)

        if direction!='auto':
            if direction=='w':
                img1, img2, lab1, lab2 = Half_w(img, lab)
            elif direction == 'h':
                img1, img2, lab1, lab2 = Half_h(img, lab)
            else:
                assert 1==0
            cv2.imwrite(imgaug_dir + '/' + img_name[index][:-4] + '_1.jpg',img1)
            cv2.imwrite(imgaug_dir + '/' + img_name[index][:-4] + '_2.jpg', img2)
            cv2.imwrite(labaug_dir + '/' + lab_name[index][:-4] + '_1.png',lab1)
            cv2.imwrite(labaug_dir + '/' + lab_name[index][:-4] + '_2.png', lab2)

        else:
            list_img=[img]
            list_lab=[lab]
            out_img=[]
            out_lab=[]
            while len(list_img)!=0:
                tmp_img=list_img[0]
                tmp_lab=list_lab[0]
                h,w=tmp_lab.shape
                if w>=2*thr:
                    img1,img2,lab1,lab2=Half_w(tmp_img,tmp_lab)
                    list_img.append(img1)
                    list_img.append(img2)
                    list_lab.append(lab1)
                    list_lab.append(lab2)
                    list_img.remove(list_img[0])
                    list_lab.remove(list_lab[0])
                elif  h>=2*thr:
                    img1,img2,lab1,lab2=Half_h(tmp_img,tmp_lab)
                    list_img.append(img1)
                    list_img.append(img2)
                    list_lab.append(lab1)
                    list_lab.append(lab2)
                    list_img.remove(list_img[0])
                    list_lab.remove(list_lab[0])
                else:
                    out_img.append(tmp_img)
                    out_lab.append(tmp_lab)
                    list_img.remove(list_img[0])
                    list_lab.remove(list_lab[0])
            for i in range(len(out_img)):
                cv2.imwrite(imgaug_dir + '/' + img_name[index][:-4] + '_'+str(i+1)+'.jpg', out_img[i])
                cv2.imwrite(labaug_dir + '/' + lab_name[index][:-4] + '_'+str(i+1)+'.png', out_lab[i])

def Filp(image, label,code):
    image = cv2.flip(image, code)
    label = cv2.flip(label, code)
    return image, label

def Gamma_correction(img,lab,gamma=1):
    corrected_img = (Unit_Normalize(img)/255)**gamma
    corrected_img = (corrected_img*255).astype(np.uint8)
    return corrected_img,lab

def Rotation_and_crop(image, label,angle,crop=None):
    h, w = label.shape
    Rotation_Matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    image = cv2.warpAffine(image, Rotation_Matrix, (w, h))
    label = cv2.warpAffine(label, Rotation_Matrix, (w, h))

    if crop:
        angle_crop=angle%180
        if angle_crop>90:
            angle_crop=180-angle_crop
        theta=angle_crop*np.pi/180
        hw_ratio=h/w
        tan_theta=np.tan(theta)
        numerator=np.cos(theta)+np.sin(theta)*tan_theta

        if h>w:
            r=hw_ratio
        else:
            r=1/hw_ratio
        denominator=r*tan_theta+1

        crop_mult=numerator/denominator
        w_crop=int(round(crop_mult*w))
        h_crop=int(round(crop_mult*h))
        x0=int((w-w_crop)/2)
        y0=int((h-h_crop)/2)
        image=image[y0:h_crop,x0:w_crop,:]
        label = label[y0:h_crop,x0:w_crop]
    return image, label

def Rotation_90(image, label,angle):
    assert angle%90==0
    h, w = label.shape
    if angle%180==0:
        Rotation_Matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        image = cv2.warpAffine(image, Rotation_Matrix, (w, h))
        label = cv2.warpAffine(label, Rotation_Matrix, (w, h))
    else:
        if h>w:
            h_pad = 0
            w_pad=int((h-w)/2)
        elif h<w:
            h_pad = int((w - h) / 2)
            w_pad = 0
        else:
            h_pad=0
            w_pad=0
        Img_pad = cv2.copyMakeBorder(image, h_pad, h_pad, w_pad, w_pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        Lab_pad = cv2.copyMakeBorder(label, h_pad, h_pad, w_pad, w_pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        Rotation_Matrix = cv2.getRotationMatrix2D((w // 2+w_pad, h // 2+h_pad), angle, 1.0)
        image_pad = cv2.warpAffine(Img_pad, Rotation_Matrix, (w+2*w_pad, h+2*h_pad))
        label_pad = cv2.warpAffine(Lab_pad, Rotation_Matrix, (w+2*w_pad, h+2*h_pad))
        image=image_pad[w_pad:w_pad+w,h_pad:h_pad+h,:]
        label = label_pad[w_pad:w_pad + w, h_pad:h_pad + h]

    return image, label

def Data_aug_RFG(img_dir,lab_dir,imgaug_dir,labaug_dir,angle_num=4,flip=True,gamma=[1],crop=True,Only_90=True):

    img_name = sorted(os.listdir(img_dir))
    lab_name = sorted(os.listdir(lab_dir))

    for index in range (len(img_name)):

        Image_file = os.path.join(img_dir, img_name[index])
        Label_file = os.path.join(lab_dir, lab_name[index])
        img = cv2.imread(Image_file, cv2.IMREAD_COLOR)
        lab = cv2.imread(Label_file, cv2.IMREAD_GRAYSCALE)

        for value in range(len(gamma)):
            if Only_90==True:
                angle_num=4
                for i in range (angle_num):
                    angle=i*360/angle_num
                    Img, Lab = Rotation_90(img, lab, angle)
                    Img,Lab=Gamma_correction(img=Img,lab=Lab,gamma=gamma[value])
                    cv2.imwrite(imgaug_dir + '/' + img_name[index][:-4] + '_R'+str(angle)+ '_G'+str(gamma[value])+'.jpg', Img)
                    cv2.imwrite(labaug_dir + '/' + lab_name[index][:-4] + '_R'+str(angle)+ '_G'+str(gamma[value])+'.png', Lab)
                    if flip:
                        Img, Lab = Filp(Img, Lab, 1)
                        cv2.imwrite(imgaug_dir + '/' + img_name[index][:-4] + '_R'+str(angle)+ '_G'+str(gamma[value]) + 'F.jpg', Img)
                        cv2.imwrite(labaug_dir + '/' + lab_name[index][:-4] + '_R'+str(angle)+ '_G'+str(gamma[value]) + 'F.png', Lab)
            else:
                for i in range (angle_num):
                    angle=i*360/angle_num
                    Img, Lab = Rotation_and_crop(img, lab, angle, crop=crop)
                    Img,Lab=Gamma_correction(img=Img,lab=Lab,gamma=gamma[value])
                    cv2.imwrite(imgaug_dir + '/' + img_name[index][:-4] + '_R'+str(angle)+ '_G'+str(gamma[value])+'.jpg', Img)
                    cv2.imwrite(labaug_dir + '/' + lab_name[index][:-4] + '_R'+str(angle)+ '_G'+str(gamma[value])+'.png', Lab)
                    if flip:
                        Img, Lab = Filp(Img, Lab, 1)
                        cv2.imwrite(imgaug_dir + '/' + img_name[index][:-4] + '_R'+str(angle)+ '_G'+str(gamma[value]) + 'F.jpg', Img)
                        cv2.imwrite(labaug_dir + '/' + lab_name[index][:-4] + '_R'+str(angle)+ '_G'+str(gamma[value]) + 'F.png', Lab)


########################Calculate CWBCE#########################################
def CWBCE_loss(input, label, balance=1.1):
    h,w=input.shape
    input=np.clip(input,0.00000001,1-0.00000001)
    label=np.clip(label, 0.00000001, 1-0.00000001)
    mask = label
    num_positive = mask.sum()
    num_negative = h * w - num_positive
    weight_positive = 1.0 * num_negative / (num_positive + num_negative)
    weight_negative = balance * num_positive / (num_positive + num_negative)
    weight = mask * weight_positive + (1 - mask) * weight_negative
    BCE = -(label * np.log(input) + (1 - label) * np.log(1 - input))
    WBCE=BCE*weight
    return WBCE.mean()

def Avg_CWBCE(Output_dir,Label_dir,balance=1.1):
    CWBCE_list=[]
    out_name = sorted(os.listdir(Output_dir))
    lab_name = sorted(os.listdir(Label_dir))
    for index in range (len(out_name)):
        out_file=os.path.join(Output_dir, out_name[index])
        label_file = os.path.join(Label_dir, lab_name[index])
        out = cv2.imread(out_file, cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)
        out=out/255
        label=label/255
        CWBCE=CWBCE_loss(out,label,balance=balance)
        CWBCE_list.append(CWBCE)
    return np.array(CWBCE_list).mean()
####################################################################


if __name__=='__main__':

    print('Processing')

    ###################Data augmentation: set the path below and run the code, all dirs should be built manually###################

    ##################Split data half#############
    # Img_dir='' #The path of training data needed to be split
    # Edge_dir=''#The path of training label needed to be split
    # Data_dir='' #The path of dir for saving training data after split
    # Label_dir=''#The path of dir for saving training labels after split
    # Data_aug_half(img_dir=Img_dir,lab_dir=Edge_dir,imgaug_dir=Data_dir,labaug_dir=Label_dir,direction='auto',thr=320)#Split data, when direction='auto', images and labels will be split half until height and width are all less than thr

    ##########Data augmentation for split data####################
    # Data_dir='' #The path of training data
    # Label_dir=''#The path of training label
    # Dataaug_dir='' #The path of dir for saving training data after augmentations
    # Labelaug_dir=''#The path of dir for saving training labels after augmentations
    # Data_aug_RFG(img_dir=Data_dir,lab_dir=Label_dir,imgaug_dir=Dataaug_dir,labaug_dir=Labelaug_dir,angle_num=4,flip=True,gamma=[1],crop=True,Only_90=True)#Rotations flips and crops, when Only_90=True, angle_num will be reset to be 4 and crop will be reset to be False
    # Add_label_to_data(Label_dir=Labelaug_dir,Data_dir=Dataaug_dir)
    # Check_name(Dataaug_dir,Labelaug_dir)

    ################################################################################################################################


    # #####################The belows are used to save model predictions#####################
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'#The device
    # device = 'cuda'
    # Test_dir=''#The path of the test data
    # Save_dir=''#The path of the dir to save predictions
    # from SDPED_Model import *#import the model
    # model = SDPED(trade_SDB=1,num_block=7)
    # checkpoint_path = ''#The path of the checkpoint
    # model=nn.DataParallel(model)
    # model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    # model=model.to(device)
    # model.eval()
    # Save_Predict(model=model,Data_dir=Test_dir,Save_pred_dir=Save_dir,device=device,pred_require=16)
    # ###################################################################################################


    # #############################To calculate the average CWBCE loss#############################
    # Output_dir = ''# Set the path of the prediction dir here
    # Label_dir=''# Set the path of the label dir here
    # CWBCE=Avg_CWBCE(Output_dir=Output_dir,Label_dir=Label_dir,balance=1.1)
    # print(CWBCE)
    # ###############################################################################################


    # #############################Use the following to save .png image to .mat type##############
    # Save_label_to_mat(Label_dir='The dir of the labels of .png type',Save_dir='The dir to save .mat type')
    # ############################################################################################
