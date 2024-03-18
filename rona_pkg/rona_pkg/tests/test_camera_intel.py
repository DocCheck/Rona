import pyrealsense2 as rs
import numpy as np
import cv2

def crop_image_from_center(im, shape):
    (width, height) = shape
    im_cx , im_cy = im.shape[1]/2 , im.shape[0]/2
    p1x , p1y = im_cx - width/2 , im_cy - height/2
    p2x , p2y = im_cx + width/2 , im_cy + height/2
    return im[int(p1y):int(p2y), int(p1x):int(p2x)]

pipeline = rs.pipeline()
config = rs.config()
colorizer = rs.colorizer()

frame_x = 1280
frame_y = 720

config.enable_stream(rs.stream.depth, frame_x, frame_y, rs.format.z16, 30)
config.enable_stream(rs.stream.color, frame_x, frame_y, rs.format.bgr8, 30)

profile = pipeline.start(config)

print("\n")
print("############################")
print(f"PRESS 'Q' TO STOP THE STREAM")
print("############################")
print("\n")

while True:
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    color_image = np.asanyarray(color_frame.get_data())
    color_image = crop_image_from_center(color_image, (960, 720))

    # get depth information of the image
    align = rs.align(rs.stream.color)
    frames = align.process(frames)
    aligned_depth_frame = frames.get_depth_frame()

    x1 = int(1280/2)-50
    x2 = int(1280/2)+50
    y1 = int(720/2)-50
    y2 = int(720/2)+50
    x =  int((x1+x2)/2)
    y =  int((y1+y2)/2)

    depth = np.asanyarray(aligned_depth_frame.get_data())
    depth = depth[y1:y2, x1:x2].astype(float)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    depth = depth * depth_scale

    depth = cv2.mean(depth)
    depth = depth[0]

    color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
    dx, dy, _= rs.rs2_deproject_pixel_to_point(color_intrin, [x,y], depth)

    #print(f"Pixel coords & depth: [{dx}:{dy}] / [{depth}]")
    colorizer = rs.colorizer()
    colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
    # plt.imshow(colorized_depth)

    cv2.imshow('frame', color_image)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()