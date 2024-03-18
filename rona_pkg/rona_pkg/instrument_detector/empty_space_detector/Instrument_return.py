import cv2
import numpy as np
from instrument_detector.object_detector.Rona_mid.utils.general import imwrite
from instrument_detector.object_detector.Rona_mid.utils.metrics import bbox_iou
from instrument_detector.object_detector.Rona_mid.utils.plots import Annotator, colors
from instrument_detector.object_detector.general_utils import bb_util
import datetime
import os
import random

import torch



class Empty_space:
    def __init__(self, es_config):
        self.slot_est_save = es_config["slot_est_save_frame"]
        self.slot_est_grid_size = es_config["slot_est_grid_size"]
        self.slot_est_frame_path = es_config["slot_est_frame_path"]
        self.return_same_slot = es_config["return_same_slot"]
        self.elevated_tray_ratio = es_config["elevated_tray_ratio"]
        self.obj_elevated_ratio = es_config["obj_elevated_ratio"]
        self.min_gripper_w = es_config["min_gripper_width"]

        if self.slot_est_save :
            if not os.path.exists(self.slot_est_frame_path):
                os.makedirs(self.slot_est_frame_path)



    def fixed_slots_estimation(self, input_img, bbox_list, classes_dict, viz=False):
        img_h, img_w, _ = np.shape(input_img)
        input_img = np.ascontiguousarray(input_img)
        n_slot_w , n_slot_h = self.slot_est_grid_size
        h_l = np.linspace(1, img_h, num=n_slot_h*2 + 1, dtype=int)
        w_l = np.linspace(1, img_w, num=n_slot_w*2 + 1, dtype=int)
        h_l = [p for i,p in enumerate(h_l) if i%2 == 1]
        w_l = [p for i,p in enumerate(w_l) if i%2 == 1]
        # slots center coordinates
        coordinate_points = [[h, w] for h in h_l for w in w_l]
        slots_bbox = [[int(c_p[1] - (img_w / n_slot_w / 2)), int(c_p[0] - (img_h / n_slot_h / 2)),
                       int(c_p[1] + (img_w / n_slot_w / 2)), int(c_p[0] + (img_h / n_slot_h / 2))] for c_p in
                      coordinate_points]
        slots_dict = {}
        for i, slot in enumerate(slots_bbox):
            slots_dict[i] = slot
        # bboxes center coordinates
        bbox_center_points = [[int(box[1]+(box[3]-box[1])/2),int(box[0]+(box[2]-box[0])/2)] for box in bbox_list]

        # maximum IOU with each slot
        for box in bbox_list:
            box1 = torch.Tensor(box[:4])
            box2 = torch.Tensor(slots_bbox)
            iou = bbox_iou(box1, box2, xywh=False).tolist()
            # box.append(np.argmax(iou))
            iou = np.concatenate(iou)
            box.append(np.where(iou > 0.0)[0].tolist())

        # calculate the empty slots list
        #occupied_list = set([box[7] for box in bbox_list])
        occupied_list = set([item for box in bbox_list for item in box[7]])
        slots_list = set(slots_dict.keys())
        red_list = list(occupied_list)
        green_list = list(slots_list.difference(set(red_list)))

        candid_coord = None
        candid_slot_id = None
        candid_coord, candid_slot_id = self.random_return_coordinate_estimation(slots_dict, green_list)
        green_list.remove(candid_slot_id)

        # Visualization
        if self.slot_est_save:
            self.visualization(input_img, slots_dict, classes_dict, coordinate_points, bbox_list, bbox_center_points,
                               green_list, red_list, candid_slot_id)
        
        if viz:
            self.visualization(input_img, slots_dict, classes_dict, coordinate_points, bbox_list, bbox_center_points,
                            green_list, red_list, candid_slot_id, vis=True)

        # convert to dictionary
        bbox_dict = self.convert_to_dict(bbox_list)

        return bbox_list, bbox_dict, slots_dict, candid_coord, input_img


    def dynamic_slots_estimation(self, input_img, bbox_list, classes_dict, obj_bbox): #
        img_h, img_w, _ = np.shape(input_img)
        r1 = set(range(1, img_w))
        r2 = [set(range(int(box[0]), int(box[2])+1)) for box in bbox_list]
        r2 = set.union(*r2)
        rr = list(r1 - r2)
        intervals_list = list(self.interval_extract(rr))
        slot_w, slot_h = int(max(self.min_gripper_w, obj_bbox[2]-obj_bbox[0])), img_h


        slots_bbox = []
        #intervals_list = [interval for interval in intervals_list if interval[1]-interval[0] > slot_w]
        for interval in intervals_list:
            if interval[1]-interval[0] > slot_w:
                sub_interval_range = sorted(range(interval[0], interval[1], slot_w))
                #sub_interval_range[-1] = interval[1]
                sub_interval_range += [interval[1]]
                sub_interval_list = [[sub_interval_range[i], sub_interval_range[i+1]] for i in range(len(sub_interval_range)-1)]
                slots_bbox.append(sub_interval_list)
            else:
                slots_bbox.append([interval])

        slots_bbox = sum(slots_bbox,[])

        for box in bbox_list:
            slots_bbox.append([int(box[0]), int(box[2])])

        slots_bbox = sorted(slots_bbox, key=lambda x: x[0])
        input_img = np.ascontiguousarray(input_img)

        # slots center coordinates
        slots_bbox = [[s_box[0], 0, s_box[1], slot_h] for s_box in slots_bbox]
        coordinate_points = [[int(box[1]+(box[3]-box[1])/2),int(box[0]+(box[2]-box[0])/2)] for box in slots_bbox]
        slots_dict = {}
        for i, slot in enumerate(slots_bbox):
            slots_dict[i] = slot

        # bboxes center coordinates
        bbox_center_points = [[int(box[1]+(box[3]-box[1])/2),int(box[0]+(box[2]-box[0])/2)] for box in bbox_list]

        # maximum IOU with each slot
        for box in bbox_list:
            box1 = torch.Tensor(box[:4])
            box2 = torch.Tensor(slots_bbox)
            iou = bbox_iou(box1, box2, xywh=False).tolist()
            box.append(np.argmax(iou))

        # calculate the empty slots list
        occupied_list = set([box[7] for box in bbox_list])
        small_list = set([key for key,value in slots_dict.items() if value[2]-value[0]<slot_w])
        slots_list = set(slots_dict.keys())
        red_list = list(set.union(small_list , occupied_list))
        green_list = list(slots_list.difference(set(red_list)))


        # Visualization
        if self.slot_est_save:
            self.visualization(input_img, slots_dict, classes_dict, coordinate_points, bbox_list, bbox_center_points, green_list, red_list)
        # convert to dictionary
        bbox_dict = self.convert_to_dict(bbox_list)

        return bbox_list, bbox_dict, slots_dict, green_list


    def convert_to_dict(self, bbox_list):
        bbox_dict = {}
        for idx, box in enumerate(bbox_list):
            bbox_dict[idx] = {"bbox": box[:4], "conf": box[4], "class_id": box[5], "est_angle": box[6], "est_slot": box[7]}
        return  bbox_dict


    def get_obj_from_slot(self, slot_id, bbox_list=None):
        if not bbox_list:
            print("Error : Object list is empty !!!")
            return None
        elif len(bbox_list[0])<8:
            print("Error : slots estimation has not been done !!!")
            return None
        else:
            obj_list = [box for box in bbox_list if box[7]==slot_id]
            if len(obj_list)==0:
                print("Slot #", str(slot_id), " contains NO object.")
            return obj_list

    def get_empty_slots(self, slot_dict, bbox_list=None):
        if not bbox_list:
            print("Error : Object list is empty !!!")
            return None
        elif len(bbox_list[0])<8:
            print("Error : slots estimation has not been done !!!")
            return None
        else:
            slot_id_list = set(slot_dict.keys())
            occupied_list = set([box[7] for box in bbox_list])
            return list(slot_id_list-occupied_list)

    def select_rnd_slot(self, candid_slots_id):
        if len(candid_slots_id) == 0:
            return None
        else:
            return candid_slots_id[random.randint(0,len(candid_slots_id)-1)]


    def returned_obj_coordinate_estimation(self, frame_sz, slots_dict, candid_slots, obj_bbox):
        if self.return_same_slot :
            slot_id = obj_bbox[7]
        else:
            slot_id = self.select_rnd_slot(candid_slots)

        obj_bb_center = bb_util.convert_bb_xywh([obj_bbox])[0]
        #print(obj_bb_center)
        obj_w, obj_h = obj_bb_center[2] , obj_bb_center[3]
        #print(frame_sz)
        [frame_w , frame_h] = frame_sz  # 800 600
        slot_bb = slots_dict[slot_id]
        returned_obj_x = slot_bb[0] + (slot_bb[2] - slot_bb[0]) / 2
        returned_obj_y = slot_bb[1] + (frame_h*self.elevated_tray_ratio) + (obj_h/2 - obj_h*self.obj_elevated_ratio)
        return [int(returned_obj_x), int(returned_obj_y)]

    def random_return_coordinate_estimation(self, slots_dict, candid_slots):
        if candid_slots == []:
            return None, None
        else:
            slot_id = self.select_rnd_slot(candid_slots)
            slot_bb = slots_dict[slot_id]
            returned_obj_x = slot_bb[0] + (slot_bb[2] - slot_bb[0]) / 2
            returned_obj_y = slot_bb[1] + (slot_bb[3] - slot_bb[1]) / 2
            return [int(returned_obj_x), int(returned_obj_y)], slot_id


    def interval_extract(self, input_list):
        input_list = sorted(set(input_list))
        range_start = previous_number = input_list[0]

        for number in input_list[1:]:
            if number == previous_number + 1:
                previous_number = number
            else:
                yield [range_start, previous_number]
                range_start = previous_number = number
        yield [range_start, previous_number]

    def visualization(self, input_img, slots_dict, classes_dict, coordinate_points, bbox_list, bbox_center_points,
                      green_list, red_list, candid_slot_id=None, vis=False):
        img_h, img_w, _ = np.shape(input_img)
        # Visualization
        if self.slot_est_save:

            shapes = np.zeros_like(input_img, np.uint8)
            cv2.rectangle(shapes, (0, 0), (img_w, int(img_h*self.elevated_tray_ratio)), (0, 50, 50), cv2.FILLED)
            mask = shapes.astype(bool)
            input_img[mask] = cv2.addWeighted(input_img, 0.5, shapes, 1 - 0.5, 0)[mask]

            #for point in coordinate_points:
            #    cv2.circle(input_img, (point[1], point[0]), 2, (100, 100, 100), -1)
            for point in bbox_center_points:
                cv2.circle(input_img, (point[1], point[0]), 2, (100, 200, 100), -1)

            if green_list is not None:
                for slot_id in green_list:
                    slot_box = slots_dict[slot_id]
                    shapes = np.zeros_like(input_img, np.uint8)
                    cv2.rectangle(shapes, (slot_box[0], slot_box[1]), (slot_box[2], slot_box[3]), (50, 0, 50), cv2.FILLED)
                    mask = shapes.astype(bool)
                    input_img[mask] = cv2.addWeighted(input_img, 0.4, shapes, 1 - 0.5, 0)[mask]

            if red_list is not None:
                for slot_id in red_list:
                    slot_box = slots_dict[slot_id]
                    shapes = np.zeros_like(input_img, np.uint8)
                    cv2.rectangle(shapes, (slot_box[0], slot_box[1]), (slot_box[2], slot_box[3]), (50, 50, 0), cv2.FILLED)
                    mask = shapes.astype(bool)
                    input_img[mask] = cv2.addWeighted(input_img, 0.4, shapes, 1 - 0.5, 0)[mask]

            if candid_slot_id is not None:
                slot_box = slots_dict[candid_slot_id]
                shapes = np.zeros_like(input_img, np.uint8)
                cv2.rectangle(shapes, (slot_box[0], slot_box[1]), (slot_box[2], slot_box[3]), (70, 250, 250), cv2.FILLED)
                mask = shapes.astype(bool)
                input_img[mask] = cv2.addWeighted(input_img, 0.4, shapes, 1 - 0.5, 0)[mask]

            annot_plot = Annotator(input_img, font_size=3, line_width=2, example=str(classes_dict))
            #for key, value in slots_dict.items():
            #    annot_plot.box_label(value, "Slot #" + str(key), color=(220,196,196))
            for box in bbox_list:
                label = classes_dict[box[5]] + f'-{box[4]:.2f}' + ", " + str(box[6]*1) + " degrees"
                annot_plot.box_label(box[:4], label, color=colors(box[5],True))
            e = datetime.datetime.now()
            frame_name = "frame_" + e.strftime("%Y%m%d_%H%M%S") + "_slots.png"
            output_path = os.path.join(self.slot_est_frame_path, frame_name)
            
            # Show or save
            if vis:
                cv2.imshow("image", input_img)
                cv2.waitKey(0) 
                cv2.destroyAllWindows()
            else:
                imwrite(output_path, input_img)
