import cv2
import numpy as np
import torch
import torchvision
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import torchsummary as summary

from preprocessing import generate_datalist, generate_dataset
from dataset import CustomDataset
from model import RotNet, AlexNet
import time

import matplotlib.pyplot as plt


def train_model():
    output_unclassified_path = "data/unclassified/"
    # model_path = "data/models/model_rot_est_128_1d_356_12022023_30.pt"
    model_path = None

    #'''
    datalist = generate_datalist(input_path="data/Orig_OPBesteck_dataset_rot_est_all/")

    dataset_img, dataset_label = generate_dataset(datalist)
    #dataset_img = dataset_img[:500]
    #dataset_label = dataset_label[:500]
    #print(dataset_label)
    X_train, X_val, y_train, y_val= train_test_split(dataset_img, dataset_label , test_size=0.05, shuffle=True)


    #datalist = generate_datalist(input_path="data/Orig_OPBesteck_dataset_rot_est_testset/2/")
    #X_val, y_val = generate_dataset(datalist)


    print(np.shape(X_train))
    print(np.shape(X_val))


    class GaussianNoise:
        def __call__(self, sample: torch.Tensor) -> torch.Tensor:

            if torch.rand(1) < 0.25 :
                noise = (torch.rand(size=(1,sample.shape[1],sample.shape[2])) < 0.8).int()
                noise = torch.where(noise == 0, 0.1, 1.0)
                if sample.shape[0]==3:
                    noise = torch.cat([noise,noise,noise],0)
            else :
                noise = torch.ones(sample.shape)
            return sample * noise


    class RndInvNoise:
        def __call__(self, sample: torch.Tensor) -> torch.Tensor:         
            if torch.rand(1) < 0.5 :
                inv = transforms.functional.invert(sample)
                px , py = np.random.randint(low=2, high=sample.shape[1]) , np.random.randint(low=2, high=sample.shape[2])
                w , h = np.random.randint(low=px, high=sample.shape[1]) , np.random.randint(low=py, high=sample.shape[2])
                sample[:,px:px+w,py:py+w] = inv[:,px:px+w,py:py+w]
            return sample 



    custom_transform = transforms.Compose([
    transforms.Grayscale(1),
    #transforms.GaussianBlur(kernel_size=(15, 15), sigma=(0.1, 2)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=(-0.1,0.1)),
    transforms.RandomResizedCrop(size=128, scale=(0.5, 1.5), ratio=(0.999, 1.001)),
    #transforms.RandomInvert(p=0.5),
    #transforms.RandomPosterize(bits=2),
    #transforms.RandomAdjustSharpness(sharpness_factor=2),

    #transforms.RandomSolarize(threshold=150.0),
    transforms.ToTensor(),
    GaussianNoise(),
    RndInvNoise(),
    ])

    train_dataset = CustomDataset(X_train, y_train, transform=custom_transform, num_class=357)
    val_dataset = CustomDataset(X_val, y_val, transform=custom_transform, num_class=357)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=256)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=256)

    #model = RotNet(num_classes=360, has_dropout=True).to('cuda:0')
    if model_path:
        model = torch.load(model_path, map_location='cuda:0')
        print("pre-trained model is loaded successfully")
    else:
        model = AlexNet(num_classes=357).to('cuda:0')
    summary.summary(model, (1, 128, 128))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    #random_input = torch.rand((1, 1, 128, 128)).to('cuda:0')
    #output = model(random_input)
    #print(output.data, torch.sum(output))

    trian_loss = {}
    val_loss = {}
    counter = 0

    start_time = time.time()
    for epoch in range(1000):
        train_losses = []
        val_losses = []
        running_loss = 0
        count = 0
        for i, inp in enumerate(train_loader):
            inputs = inp[0]
            inputs = inputs.to('cuda:0')
            #print("### inputs shape : ", inputs.shape)
            labels = inp[1]
            labels = labels.to('cuda:0')
            optimizer.zero_grad()

            #if epoch == 0 :
            #    for ii in range(inputs.size(0)):
            #        if counter < 5000 :
            #            save_image(inputs[ii], output_unclassified_path+"img_"+str(counter)+"_"+str(labels[ii])+".png")
            #            counter += 1 

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            #print(torch.max(labels.data, 1)[1])
            #print(torch.max(outputs.data, 1)[1])
            #print(outputs)
            #print(labels)
            #print("loss : ", loss)
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            count += 1

        if epoch % 5 == 0:
            for i, inp in enumerate(val_loader):
                inputs = inp[0]
                inputs = inputs.to('cuda:0')
                labels = inp[1]
                labels = labels.to('cuda:0')
                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())

            trian_loss[epoch] = np.mean(train_losses)
            val_loss[epoch] = np.mean(val_losses)
            print('Epoch', epoch, ': train loss =', np.mean(train_losses), ', val loss =', np.mean(val_losses))

    end_time = time.time()
    print('Training Done in', end_time - start_time, 'seconds')

    fig = plt.figure(figsize=(20, 20))
    plt.plot(list(trian_loss.keys()), list(trian_loss.values()))
    plt.plot(list(val_loss.keys()), list(val_loss.values()))
    plt.savefig("train_val_loss_model_rot_est_128_1d_357_16022023_30.png")
    plt.close()


    correct_train = 0
    total_train = 0

    correct_val = 0
    total_val = 0

    val_output = []

    with torch.no_grad():
        for data in train_loader:
            tensor = data[0]
            tensor = tensor.to('cuda:0')
            label = data[1]
            label = label.to('cuda:0')
            outputs = model(tensor)

            _, predicted = torch.max(outputs.data, 1)
            #print(outputs)
            #print(label)
            #_, label = torch.max(label.data, 1)
            #print(predicted)
            #print(label)
            total_train += tensor.size(0)
            correct_train += (predicted == label).sum().item()

        for data in val_loader:
            tensor = data[0]
            tensor = tensor.to('cuda:0')
            label = data[1]
            label = label.to('cuda:0')
            outputs = model(tensor)

            _, predicted = torch.max(outputs.data, 1)
            #_, label = torch.max(label.data, 1)
            print(predicted)
            print(label)
            val_output.append(predicted)
            total_val += tensor.size(0)
            correct_val += (predicted == label).sum().item()

    print('Accuracy on Train Data :', 100 * (correct_train / total_train), '%')
    print('Accuracy on Validation Data :', 100 * (correct_val / total_val), '%')
    print('Validation length :', len(val_output) , correct_val, total_val)



    print("############### saving the model ######################")
    model_path = "data/models/instrument_detector_model.pt"
    torch.save(model, model_path)
    #'''

    ############################# Inference ##########################
    '''
    class GaussianNoise:
        def __call__(self, sample: torch.Tensor) -> torch.Tensor:
            # mu = sample.mean()
            # print("############" , mu)
            # snr = np.random.randint(low=40, high=80)
            # sigma = mu / snr
            # noise = torch.normal(torch.zeros(sample.shape), sigma)
            # print(sample.shape)
            # noise = torch.randint(0, 2, (1,sample.shape[1],sample.shape[2]))
            if torch.rand(1) < 0.5:
                noise = (torch.rand(size=(1, sample.shape[1], sample.shape[2])) < 0.9).int()
                noise = torch.where(noise == 0, 0.2, 1.0)
                if sample.shape[0] == 3:
                    noise = torch.cat([noise, noise, noise], 0)
            else:
                noise = torch.ones(sample.shape)
            # print("############", noise.shape)
            # print(sample * noise)
            return sample * noise

    class RndInvNoise:
        def __call__(self, sample: torch.Tensor) -> torch.Tensor:
            mu = sample.mean()
            # print("############" , mu)
            snr = np.random.randint(low=40, high=80)
            sigma = mu / snr
            # noise = torch.normal(torch.zeros(sample.shape), sigma)
            # print(sample.shape)
            # noise = torch.randint(0, 2, (1,sample.shape[1],sample.shape[2]))

            if torch.rand(1) < 0.5:
                inv = transforms.functional.invert(sample)
                px, py = np.random.randint(low=2, high=sample.shape[1]), np.random.randint(low=2, high=sample.shape[2])
                w, h = np.random.randint(low=px, high=sample.shape[1]), np.random.randint(low=py, high=sample.shape[2])
                sample[:, px:px + w, py:py + w] = inv[:, px:px + w, py:py + w]
            return sample

    datalist = generate_datalist(input_path="data/Orig_OPBesteck_dataset_rot_est_all/")
    X_test, y_test = generate_dataset(datalist)
    # X_test = X_test[:2000]
    # y_test = y_test[:2000]

    custom_transform_inference = transforms.Compose([
        transforms.Grayscale(1),
        # transforms.GaussianBlur(kernel_size=(15, 15), sigma=(0.1, 2)),
        # transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.3, hue=(-0.1,0.1)),
        transforms.RandomResizedCrop(size=128, scale=(0.95, 1.05), ratio=(0.999, 1.001)),
        # transforms.RandomInvert(p=0.5),
        # transforms.RandomPosterize(bits=2),
        # transforms.RandomAdjustSharpness(sharpness_factor=2),

        # transforms.RandomSolarize(threshold=150.0),
        transforms.ToTensor(),
        # GaussianNoise(),
        # RndInvNoise(),
    ])

    test_dataset = CustomDataset(X_test, y_test, transform=custom_transform_inference, num_class=357)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=256)

    model_path = "data/models/instrument_detector_model.pt"
    device = torch.device('cpu')
    loaded_model = torch.load(model_path, map_location=device)
    correct_val = 0
    total_val = 0
    val_output = []
    with torch.no_grad():
        counter = 0
        for data in test_loader:
            tensor = data[0]
            # tensor = tensor.to('cuda:0')
            label = data[1]
            # label = label.to('cuda:0')
            outputs = loaded_model(tensor)

            _, predicted = torch.max(outputs.data, 1)
            __, predicted_list = torch.sort(outputs.data, 1, descending=True)
            # print(predicted_list[0])
            # print(__[0])
            # print(predicted_list.size)
            # exit(0)
            # predicted = predicted_list[0]
            # _, label = torch.max(label.data, 1)
            # print(predicted)
            # print(label)
            val_output.append(predicted)
            total_val += tensor.size(0)
            correct_val += (torch.abs(predicted - label) < 3).sum().item()

            for i in range(tensor.size(0)):
                if torch.abs(predicted[i] - label[i]) > 3:
                    save_image(tensor[i],
                               output_unclassified_path + "img_" + str(counter) + "_" + str(label[i]) + "_" + str(
                                   predicted[i]) + ".png")
                    counter += 1
                    print(predicted_list[i])
                    print(__[i])

    print('Accuracy on Validation Data :', 100 * (correct_val / total_val), '%')
    print('Validation length :', len(val_output), correct_val, total_val)
    '''


if __name__ == '__main__':
    train_model()