#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 12:08:25 2020

@author: laakom
"""

import numpy as np
import cv2
import math
import random
import pickle 
from tqdm import tqdm
import os
# Use data augmentation?
AUGMENTATION = True
# Rotation angle
AUGMENTATION_ANGLE = 60
# Patch scale
AUGMENTATION_SCALE = [0.1, 1.0]
# Random left-right flip?
AUGMENTATION_FLIP_LEFTRIGHT = True
# Random top-down flip?
AUGMENTATION_FLIP_TOPDOWN = False
# Color rescaling?
AUGMENTATION_COLOR = 0.8
# Cross-channel terms
AUGMENTATION_COLOR_OFFDIAG = 0.0
# Augment Gamma?
AUGMENTATION_GAMMA = 0.0
# Augment using a polynomial curve?
USE_CURVE = False
# Apply different gamma and curve to left/right halves?
SPATIALLY_VARIANT = False




















def rotate_image(image, angle):
  """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

  # Get the image size
  # No that's not an error - NumPy stores image matricies backwards
  image_size = (image.shape[1], image.shape[0])
  image_center = tuple(np.array(image_size) / 2)

  # Convert the OpenCV 3x2 rotation matrix to 3x3
  rot_mat = np.vstack(
      [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]])

  rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

  # Shorthand for below calcs
  image_w2 = image_size[0] * 0.5
  image_h2 = image_size[1] * 0.5

  # Obtain the rotated coordinates of the image corners
  rotated_coords = [
      (np.array([-image_w2, image_h2]) * rot_mat_notranslate).A[0],
      (np.array([image_w2, image_h2]) * rot_mat_notranslate).A[0],
      (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
      (np.array([image_w2, -image_h2]) * rot_mat_notranslate).A[0]
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
  trans_mat = np.matrix([[1, 0, int(new_w * 0.5 - image_w2)],
                         [0, 1, int(new_h * 0.5 - image_h2)], [0, 0, 1]])

  # Compute the tranform for the combined rotation and translation
  affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

  # Apply the transform
  result = cv2.warpAffine(
      image, affine_mat, (new_w, new_h), flags=cv2.INTER_LINEAR)

  return result


def largest_rotated_rect(w, h, angle):
  """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
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

  return (bb_w - 2 * x, bb_h - 2 * y)


def crop_around_center(image, width, height):
  """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

  image_size = (image.shape[1], image.shape[0])
  image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

  if (width > image_size[0]):
    width = image_size[0]

  if (height > image_size[1]):
    height = image_size[1]

  x1 = int(image_center[0] - width * 0.5)
  x2 = int(image_center[0] + width * 0.5)
  y1 = int(image_center[1] - height * 0.5)
  y2 = int(image_center[1] + height * 0.5)

  return image[y1:y2, x1:x2]


def rotate_and_crop(image, angle):
  image_width, image_height = image.shape[:2]
  image_rotated = rotate_image(image, angle)
  image_rotated_cropped = crop_around_center(image_rotated,
                                             *largest_rotated_rect(
                                                 image_width, image_height,
                                                 math.radians(angle)))
  return image_rotated_cropped











def augment_data(train_size,Images,groundtruths, patch_size,save_path):
               
    number_of_train_gt = len(Images);
    
    
    patches_per_image = int(train_size/number_of_train_gt);

    count = 0 
    aug_groundtruth = {}
    for img in tqdm(Images):
        image = (cv2.resize(cv2.imread(img,-1),(0,0), fx=0.75, fy=0.75) *1.0 / 65535.0 ).astype('float32')
        gt = groundtruths[img]        
        for j in range (0, patches_per_image):
            name = save_path + '/' + str(count)+ '.png'
            aug_img , aug_illum = augment(image, gt,patch_size)
            #save image
            cv2.imwrite( name, (aug_img*255.0).astype('uint8') ) 
            aug_groundtruth[os.path.realpath(name)] = aug_illum           
            count += 1 
    output = open(save_path + '/ground_truth.pkl', 'wb')
    pickle.dump(aug_groundtruth, output)
    output.close()
    

def augment(ldr, illum,patch_size):
  angle = (random.random() - 0.5) * AUGMENTATION_ANGLE
  scale = math.exp(random.random() * math.log(
      AUGMENTATION_SCALE[1] / AUGMENTATION_SCALE[0])) * AUGMENTATION_SCALE[0]
  s = int(round(min(ldr.shape[:2]) * scale))
  s = min(max(s, 10), min(ldr.shape[:2]))
  start_x = random.randrange(0, ldr.shape[0] - s + 1)
  start_y = random.randrange(0, ldr.shape[1] - s + 1)
  # Left-right flip?
  flip_lr = random.randint(0, 1)
  # Top-down flip?
  flip_td = random.randint(0, 1)
  color_aug = np.zeros(shape=(3, 3))
  for i in range(3):
    color_aug[i, i] = 1 + random.random(
    ) * AUGMENTATION_COLOR - 0.5 * AUGMENTATION_COLOR
    for j in range(3):
      if i != j:
        color_aug[i, j] = (random.random() - 0.5) * AUGMENTATION_COLOR_OFFDIAG
  maxin = np.max(ldr)  
  def crop(img, illumination,patch_size):
    if img is None:
      return None
    img = img[start_x:start_x + s, start_y:start_y + s]
    img = rotate_and_crop(img, angle)
    img = cv2.resize(img, patch_size)
    if AUGMENTATION_FLIP_LEFTRIGHT and flip_lr:
      img = img[:, ::-1]
    if AUGMENTATION_FLIP_TOPDOWN and flip_td:
      img = img[::-1, :]

    img = img.astype(np.float32)
    new_illum = np.zeros_like(illumination)
    # RGB -> BGR
    illumination = illumination[::-1]
    for i in range(3):
      for j in range(3):
        new_illum[i] += illumination[j] * color_aug[i, j]
    if AUGMENTATION_COLOR_OFFDIAG > 0:
      # Matrix mul, slower
      new_image = np.zeros_like(img)
      for i in range(3):
        for j in range(3):
          new_image[:, :, i] += img[:, :, j] * color_aug[i, j]
    else:
      img *= np.array(
          [[[color_aug[0][0], color_aug[1][1], color_aug[2][2]]]],
          dtype=np.float32)
      new_image = img
    new_image = np.clip(new_image, 0, maxin)


    new_illum = np.clip(new_illum, 0.01, 100)

    return new_image, new_illum[::-1]

  return crop(ldr, illum,patch_size)












def augment_col(ldr, illum):
  color_aug = np.zeros(shape=(3, 3))
  for i in range(3):
    color_aug[i, i] = 1 + random(
    ) * AUGMENTATION_COLOR - 0.5 * AUGMENTATION_COLOR
    for j in range(3):
      if i != j:
        color_aug[i, j] = (random() - 0.5) * AUGMENTATION_COLOR_OFFDIAG
  maxin = np.max(ldr)  

  new_illum = np.zeros_like(illum)

  for i in range(3):
   for j in range(3):
     new_illum[i] += illum[j] * color_aug[i, j]

  ldr *= np.array(
      [[[color_aug[0][0], color_aug[1][1], color_aug[2][2]]]],
      dtype=np.float32)
  ldr = np.clip(ldr, 0, maxin)

  new_illum = np.clip(new_illum, 0.01, np.max(new_illum) )
  #new_illum /= np.linalg.norm(new_illum,2)
  return ldr, new_illum

def create_lut(f,resolution):
    num_samples = resolution
    lut = np.array(  [f(x)  for x in np.linspace(0,1,num_samples)], dtype = np.float32  )
    return lambda x: np.take(lut, x.astyple('int32'))