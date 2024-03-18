from instrument_detector import general_config


def load_runner_config():
    runner_config = {}
    runner_config["runner_on_demand"] = general_config.runner_on_demand
    runner_config["runner_period"] = general_config.runner_period
    return runner_config

def load_camera_config():
    cam_config = {}
    cam_config["cam_port"] = general_config.cam_port
    cam_config["cam_frame_width"] = general_config.cam_frame_width
    cam_config["cam_frame_height"] = general_config.cam_frame_height
    cam_config["cam_save_frame"] = general_config.cam_save_frame
    cam_config["cam_frame_path"] = general_config.cam_frame_path
    cam_config["cam_frame_output_sz"] = general_config.cam_center_crop_size
    return cam_config

def load_ori_est_config():
    ooe_config = {}
    ooe_config["ooe_gaussian_blur_filter"] = general_config.ooe_gaussian_blur_filter
    ooe_config["ooe_binary_thr_min"] = general_config.ooe_binary_thr_min
    ooe_config["ooe_binary_thr_max"] = general_config.ooe_binary_thr_max
    ooe_config["ooe_crop_padding"] = general_config.ooe_crop_padding

    ooe_config["model_path"] = general_config.ooe_rotnet_model_path
    ooe_config["padding_size"] = general_config.ooe_rotnet_padding
    ooe_config["image_size"] = general_config.ooe_rotnet_image_size
    ooe_config["device"] = general_config.ooe_device
    return ooe_config

def load_mid_config():
    model_config = {}
    model_config["model_path"] = general_config.model_path
    model_config["image_size"] = general_config.image_size
    model_config["confidence_threshold"] = general_config.confidence_threshold
    model_config["iou_threshold"] = general_config.iou_threshold
    model_config["classes"] = general_config.classes
    model_config["agnostic_nms"] = general_config.agnostic_nms
    model_config["maximum_detection"] = general_config.maximum_detection
    model_config["device"] = general_config.device
    return model_config


def load_slot_est_config():
    se_config = {}
    se_config["slot_est_save_frame"] = general_config.slot_est_save_frame
    se_config["slot_est_grid_size"] = general_config.slot_est_grid_size
    se_config["slot_est_frame_path"] = general_config.slot_est_frame_path
    se_config["return_same_slot"] = general_config.return_same_slot
    se_config["elevated_tray_ratio"] = general_config.elevated_tray_ratio
    se_config["obj_elevated_ratio"] = general_config.obj_elevated_ratio
    se_config["min_gripper_width"] = general_config.min_gripper_width
    return se_config