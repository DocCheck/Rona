import numpy as np
import cv2
import math
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt

def convert_yolo_bb_to_abs(annot_list,img_shape):
    [width, height] = img_shape
    abs_annot_list = []
    for annot in annot_list:
        abs_annot = [0,0,0,0]
        abs_annot[0] = (annot[0] * width) - ((annot[2] * width) / 2)
        abs_annot[1] = (annot[1] * height) - ((annot[3] * height) / 2)
        abs_annot[2] = annot[2] * width
        abs_annot[3] = annot[3] * height
        abs_annot_list.append(abs_annot)
    return abs_annot_list

def crop_obj_squared(img, bbox_l, pad_size=6):
    (img_height, img_width, _) = np.shape(img)
    print(bbox_l)
    new_l = int(max(bbox_l[2], bbox_l[3])) + (2 * pad_size)
    [d_width, d_height] = [new_l, new_l]
    center_x, center_y = bbox_l[0] + bbox_l[2]/2 , bbox_l[1] + bbox_l[3]/2
    bb_cx, bb_cy, bb_w, bb_h = int(center_x-(d_width/2)), int(center_y-(d_height/2)), int(d_width), int(d_height)
    if (d_height < bb_h) or (d_width < bb_w):
        return (img, bbox_l)
    else:
        img = img[bb_cy:bb_cy+d_height, bb_cx:bb_cx+d_width, :]
        #print("image_size after crop :", np.shape(img))
        return img

def crop_obj_double_bb(img, bbox_l, pad_size=6):
    (img_height, img_width, _) = np.shape(img)
    print(bbox_l)
    new_l = int(max(bbox_l[2], bbox_l[3]))
    [d_width, d_height] = [new_l*2, new_l*2]
    center_x, center_y = bbox_l[0] + bbox_l[2]/2 , bbox_l[1] + bbox_l[3]/2
    bb_cx, bb_cy, bb_w, bb_h = int(center_x-(d_width/2)), int(center_y-(d_height/2)), int(d_width), int(d_height)
    if (d_height < bb_h) or (d_width < bb_w):
        return (img, bbox_l)
    else:
        img = img[bb_cy:bb_cy+d_height, bb_cx:bb_cx+d_width, :]
        #print("image_size after crop :", np.shape(img))
        return img,

def crop_obj_with_padding(img, bbox_l, desired_size, pad_size=6):
    (img_height, img_width, _) = np.shape(img)
    new_l = int(max(bbox_l[2], bbox_l[3])) + (2 * pad_size)
    [d_width, d_height] = [int(bbox_l[2]) + (2 * pad_size), int(bbox_l[3]) + (2 * pad_size)]
    center_x, center_y = bbox_l[0] + bbox_l[2] / 2, bbox_l[1] + bbox_l[3] / 2
    bb_cx, bb_cy, bb_w, bb_h = int(center_x - (d_width / 2)), int(center_y - (d_height / 2)), int(d_width), int(
        d_height)
    img = img[bb_cy:bb_cy + d_height, bb_cx:bb_cx + d_width, :]

    #print(img.shape)

    old_size = img.shape[:2]  # old_size is in (height, width) format
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    img = cv2.resize(img, (new_size[1], new_size[0]))
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [160, 160, 160]
    new_im = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REPLICATE, value=color) #cv2.BORDER_CONSTANT BORDER_REPLICATE
    #print(new_im.shape)
    return new_im


def resize_with_padding(img, desired_size, resize=True):

    old_size = img.shape[:2]  # old_size is in (height, width) format
    ratio = float(desired_size / max(old_size))
    new_size = tuple([int(x * ratio) for x in old_size])
    if resize:
        img = cv2.resize(img, (new_size[1], new_size[0]))
    #else:
    #    ratio = 1.0
    #    new_size = tuple([int(x * ratio) for x in old_size])

    #print("__________ : " ,new_size)
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    #print(new_im.shape)
    return new_im

def cropp_center(img, desired_size):
    (d_width, d_height) = desired_size
    image_size = (img.shape[0], img.shape[1])
    image_center = tuple(np.array(image_size) / 2)
    #print(d_width , d_height)
    #print(image_center)
    img = img[int(image_center[0] - (d_width / 2)):int(image_center[0] + (d_width / 2)),
    int(image_center[1] - (d_height / 2)):int(image_center[1] + (d_height / 2))]

    return img





def bg_remove(img):
    # parameters
    blur = 21
    canny_low = 15
    canny_high = 150
    min_area = 0.0005
    max_area = 0.95
    dilate_iter = 10
    erode_iter = 10
    mask_color = (0.0, 0.0, 0.0)

    # Convert image to grayscale
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Canny Edge Detection
    edges = cv2.Canny(image_gray, canny_low, canny_high)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)
    print(np.shape(edges))
    contour_info = [(c, cv2.contourArea(c),) for c in cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]]
    # Get the area of the image as a comparison
    image_area = img.shape[0] * img.shape[1]
    # calculate max and min areas in terms of pixels
    max_area = max_area * image_area
    min_area = min_area * image_area

    # Set up mask with a matrix of 0's
    mask = np.zeros(edges.shape, dtype=np.uint8)

    # Go through and find relevant contours and apply to mask
    for contour in contour_info:
        # Instead of worrying about all the smaller contours, if the area is smaller than the min, the loop will break
        if contour[1] > min_area and contour[1] < max_area:
            # Add contour to mask
            mask = cv2.fillConvexPoly(mask, contour[0], (255))

    # use dilate, erode, and blur to smooth out the mask
    mask = cv2.dilate(mask, None, iterations=dilate_iter)
    mask = cv2.erode(mask, None, iterations=erode_iter)
    print(np.shape(mask))
    mask = cv2.GaussianBlur(mask, (blur, blur), 0)

    # Ensures data types match up
    mask_stack = np.dstack([mask] * 3)
    mask_stack = mask_stack.astype('float32') / 255.0
    frame = img.astype('float32') / 255.0

    # Blend the image and the mask
    masked = (mask_stack * frame) + ((1 - mask_stack) * mask_color)
    masked = (masked * 255).astype('uint8')

    return masked










def rotate(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    Source: https://github.com/d4nst/RotNet/tree/master
    license: MIT License
    """
    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    #print(np.shape(image))
    #print(np.shape(affine_mat))
    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result


def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.
    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow
    Converted to Python by Aaron Snoswell
    Source: https://github.com/d4nst/RotNet/tree/master
    license: MIT License
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )

def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    Source: https://github.com/d4nst/RotNet/tree/master
    license: MIT License
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]

def crop_largest_rectangle(image, angle, height, width):
    """
    Crop around the center the largest possible rectangle
    found with largest_rotated_rect.
    Source: https://github.com/d4nst/RotNet/tree/master
    license: MIT License
    """
    return crop_around_center(
        image,
        *largest_rotated_rect(
            width,
            height,
            math.radians(angle)
        )
    )

def generate_rotated_image(image, angle, size=None, crop_center=False,
                           crop_largest_rect=False):
    """
    Generate a valid rotated image for the RotNetDataGenerator. If the
    image is rectangular, the crop_center option should be used to make
    it square. To crop out the black borders after rotation, use the
    crop_largest_rect option. To resize the final image, use the size
    option.
    Source: https://github.com/d4nst/RotNet/tree/master
    license: MIT License
    """
    height, width = image.shape[:2]
    if crop_center:
        if width < height:
            height = width
        else:
            width = height

    image = rotate(image, angle)

    if crop_largest_rect:
        image = crop_largest_rectangle(image, angle, height, width)

    if size:
        image = cv2.resize(image, size)

    return image



def display_image_grid(images, n=10, angles=None, string_="N"):
    '''
    This function visualizes the given images in a grid with labels
    Source: https://www.kaggle.com/code/alibalapour/rotation-prediction-de-skewing-text-in-images/notebook
    license: Apache 2.0
    '''
    fig = plt.figure(figsize=(20, 20))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(n, n),
                     axes_pad=0.25,
                     )

    i = 0
    for ax, im in zip(grid, images):
        ax.imshow(im, cmap='gray');
        ax.set_xticks([])
        ax.set_yticks([])
        if angles is not None:
            angle = angles[i] - 5
            ax.set_title(label=str(angle))
        i += 1

    plt.savefig("result1"+str(string_)+".png")
    plt.close()
    #plt.show()
