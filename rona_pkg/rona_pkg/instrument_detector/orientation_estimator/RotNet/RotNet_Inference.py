import torch
from torchvision.utils import save_image
import numpy as np
import cv2
import copy
import datetime

from instrument_detector.orientation_estimator.RotNet.utils import crop_obj_with_padding


class RotNet:
    def __init__(self, model_config):
        self.model_path = model_config["model_path"]
        self.imgsz = model_config["image_size"]
        self.device = torch.device(model_config["device"])
        self.padding = model_config["padding_size"]


    def load_model(self):
        device = torch.device(self.device)
        self.model = torch.load(self.model_path, map_location=device)
        return self.model


    def predict(self, img, input_bb_list):
        #img = cv2.GaussianBlur(img, (15, 15), cv2.BORDER_DEFAULT)
        bb_list = copy.deepcopy(input_bb_list)
        if len(bb_list) == 0:
            print("No instrument has been detected !!!")
            return None

        #convert x1,y1,x2,y2 to x1,y1,w,h
        for pred in bb_list:
            pred[2] -= pred[0]
            pred[3] -= pred[1]
        #print("############### : " , input_bb_list)
        obj_list = []
        for pred in bb_list:
            bbox = pred[:4]
            #cropped_img = crop_obj_squared(img, bbox, pad_size=self.padding)
            #cropped_img = cv2.resize(cropped_img, self.imgsz)
            cropped_img = crop_obj_with_padding(img, bbox, desired_size=self.imgsz[0], pad_size=self.padding)
            #exit(0)
            #cropped_img = bg_remove(cropped_img)
            cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
            cropped_img = cropped_img / 255

            cropped_img = np.expand_dims(cropped_img, axis=0)
            #print(np.shape(cropped_img))
            obj_list.append(cropped_img)

        #print(np.shape(obj_list))
        #preprocess = transforms.Compose([transforms.ToTensor()])
        #exit(0)
        self.model.eval()
        with torch.no_grad():
            #if len(obj_list) == 1:
            #    obj_list = obj_list[0]
            tensor = torch.tensor(np.array(obj_list)).float()
            #print(tensor.shape)
            # tensor = tensor.to('cuda:0')

            outputs = self.model(tensor)

            _, prediction = torch.max(outputs.data, 1)
            e = datetime.datetime.now()
            save_image(tensor, "output___" + str(pred[5]) + "_" + str(prediction*1) + e.strftime("_%Y_%H%M%S") + ".png")
            # _, label = torch.max(label.data, 1)
            #print(prediction)

        prediction = prediction.tolist()
        for i, pred in enumerate(input_bb_list):
            pred.append(prediction[i])

        #print("############### : " , input_bb_list)
        return input_bb_list


