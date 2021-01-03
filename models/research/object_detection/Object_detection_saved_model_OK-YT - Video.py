#!/usr/bin/env python
# coding: utf-8

# 
# Object Detection From Saved Model (TensorFlow 2)
# =====================================
# 

# In[1]:


import os
import time
import tensorflow as tf
print (tf.__version__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# In[2]:


IMAGE_PATHS = ['test_images/image1.jpg',
               'test_images/image2.jpg',
              'test_images/image3.jpg',
              'test_images/image4.jpg',
              'test_images/image5.jpg']


# In[3]:


import cv2

cap = cv2.VideoCapture('video/1.mp4')


# In[4]:


#IMAGE_PATHS


# In[5]:


PATH_TO_LABELS='data/mscoco_label_map.pbtxt'


# In[6]:


import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils


# In[7]:


PATH_TO_SAVED_MODEL='models/ssd_mobilenet_v2_coco_2018_03_29/saved_model'


# In[8]:


print('Loading model...', end='')
start_time = time.time()
print (start_time)

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
detect_fn = detect_fn.signatures['serving_default']

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))


# Load label map data (for plotting)
# 
# 
# 

# In[9]:


category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,use_display_name=True)



import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

def load_image_into_numpy_array(path):
    return np.array(Image.open(path))


# In[12]:


for image_path in IMAGE_PATHS:
    print(f'Running inference for {image_path}...')
    image_np = load_image_into_numpy_array(image_path)

    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn(input_tensor)
     
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    
    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np,detections['detection_boxes'],detections['detection_classes'], detections['detection_scores'],
          category_index,use_normalized_coordinates=True,
          max_boxes_to_draw=200,min_score_thresh=.30,agnostic_mode=False)

    plt.figure()
    plt.imshow(image_np)
    print("Done\n" )
plt.show()


# In[13]:


os.listdir()


# In[14]:



while 1:
    _,img = cap.read()
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    print(f'Running inference for {img}...')
    #image_np = load_image_into_numpy_array(image_path)

    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(img)

    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    print (detections.keys())

    num_detections = int(detections.pop('num_detections'))
    print ("Number of Objects in the image = ",num_detections)
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}

    detections['num_detections'] = num_detections
    print ("Number of Objects in the image = ",num_detections)
    print (detections.items())
    print (detections.keys())
    print ("Classes in the image are : " ,detections['detection_classes'])

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    #image_np_with_detections = image_np.copy()
    #print (image_np_with_detections)


    ### VISUALIZATION ON THE IMAGE
    final_img = viz_utils.visualize_boxes_and_labels_on_image_array(
          img,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False)


    print('Done\n' )

    final_img = cv2.cvtColor(final_img,cv2.COLOR_RGB2BGR)
    cv2.imshow('img',final_img)

    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# sphinx_gallery_thumbnail_number = 2


# In[ ]:





# In[ ]:




