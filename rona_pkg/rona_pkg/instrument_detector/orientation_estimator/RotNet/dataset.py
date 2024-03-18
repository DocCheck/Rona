import torch
import torchvision
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import one_hot
from torchvision import transforms
from PIL import Image
import numpy as np
import datetime


class CustomDataset(Dataset):
    def __init__(self, img_list, label_list, transform=None, num_class=360):
        """
        Args:
            df (pd.DataFrame): DataFrame of image names and labels.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_list = img_list
        self.label_list = label_list
        self.transform = transform
        self.num_class = num_class

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.img_list[idx]
        # image = torchvision.transforms.ToTensor()(image)
        # print("########### ", image.shape)
        image = Image.fromarray(np.uint8(image)).convert('RGB')
        label = int(self.label_list[idx] / 1)  # + 180

        # output_unclassified_path = "data/input_sim/"
        # trans = transforms.Compose([transforms.ToTensor()])
        # if label == 0 or label == 1 :
        #    e = datetime.datetime.now()
        #    save_image(trans(image), output_unclassified_path+ "output___" +str(label)+ e.strftime("_%Y_%H%M%S") + ".png")

        if self.transform:
            image = self.transform(image)

        # print(torch.max(image) , torch.min(image))
        # label = int(self.label_list[idx] / 1) #+ 180
        # print("label : ", label)
        # label = one_hot(torch.tensor(label) , self.num_class).type(torch.float32)
        # print("label : ", label)

        return image, label

