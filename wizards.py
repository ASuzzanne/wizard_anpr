#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os


# In[2]:

CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

# In[3]:


paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
    'APIMODEL_PATH': os.path.join('Tensorflow','models'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
    'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
    'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
    'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
    'PROTOC_PATH':os.path.join('Tensorflow','protoc')
 }


# In[4]:


files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}


# In[5]:


for path in paths.values():
    if not os.path.exists(path):
        if os.name == 'posix':
            get_ipython().system('mkdir -p {path}')
        if os.name == 'nt':
            get_ipython().system('mkdir {path}')


# # 1. Download TF Models Pretrained Models from Tensorflow Model Zoo and Install TFOD

# In[6]:


if os.name=='nt':
    get_ipython().system('pip install wget')
    import wget


# In[7]:


if not os.path.exists(os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection')):
    get_ipython().system("git clone https://github.com/tensorflow/models {paths['APIMODEL_PATH']}")


# In[8]:


# Install Tensorflow Object Detection 
if os.name=='posix':  
    get_ipython().system('apt-get install protobuf-compiler')
    get_ipython().system('cd Tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=. && cp object_detection/packages/tf2/setup.py . && python -m pip install . ')
    
if os.name=='nt':
    url="https://github.com/protocolbuffers/protobuf/releases/download/v3.15.6/protoc-3.15.6-win64.zip"
    wget.download(url)
    get_ipython().system("move protoc-3.15.6-win64.zip {paths['PROTOC_PATH']}")
    get_ipython().system("cd {paths['PROTOC_PATH']} && tar -xf protoc-3.15.6-win64.zip")
    os.environ['PATH'] += os.pathsep + os.path.abspath(os.path.join(paths['PROTOC_PATH'], 'bin'))   
    get_ipython().system('cd Tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=. && copy object_detection\\\\packages\\\\tf2\\\\setup.py setup.py && python setup.py build && python setup.py install')
    get_ipython().system('cd Tensorflow/models/research/slim && pip install -e . ')


# In[9]:


VERIFICATION_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'builders', 'model_builder_tf2_test.py')
# Verify Installation
get_ipython().system('python {VERIFICATION_SCRIPT}')


# In[ ]:





# In[10]:


import object_detection


# In[11]:


if os.name =='posix':
    get_ipython().system('wget {PRETRAINED_MODEL_URL}')
    get_ipython().system("mv {PRETRAINED_MODEL_NAME+'.tar.gz'} {paths['PRETRAINED_MODEL_PATH']}")
    get_ipython().system("cd {paths['PRETRAINED_MODEL_PATH']} && tar -zxvf {PRETRAINED_MODEL_NAME+'.tar.gz'}|")
if os.name == 'nt':
    wget.download(PRETRAINED_MODEL_URL)
    get_ipython().system("move {PRETRAINED_MODEL_NAME+'.tar.gz'} {paths['PRETRAINED_MODEL_PATH']}")
    get_ipython().system("cd {paths['PRETRAINED_MODEL_PATH']} && tar -zxvf {PRETRAINED_MODEL_NAME+'.tar.gz'}")


# # 2. Create Label Map

# In[12]:


labels = [{'name':'licence', 'id':1}]

with open(files['LABELMAP'], 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')


# # 3. Create TF records

# In[13]:


if not os.path.exists(files['TF_RECORD_SCRIPT']):
    get_ipython().system("git clone https://github.com/nicknochnack/GenerateTFRecord {paths['SCRIPTS_PATH']}")


# In[14]:


get_ipython().system("python {files['TF_RECORD_SCRIPT']} -x {os.path.join(paths['IMAGE_PATH'], 'train')} -l {files['LABELMAP']} -o {os.path.join(paths['ANNOTATION_PATH'], 'train.record')} ")
get_ipython().system("python {files['TF_RECORD_SCRIPT']} -x {os.path.join(paths['IMAGE_PATH'], 'test')} -l {files['LABELMAP']} -o {os.path.join(paths['ANNOTATION_PATH'], 'test.record')} ")


# # 4. Copy Model Config to Training Folder

# In[15]:


if os.name =='posix':
    get_ipython().system("cp {os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'pipeline.config')} {os.path.join(paths['CHECKPOINT_PATH'])}")
if os.name == 'nt':
    get_ipython().system("copy {os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'pipeline.config')} {os.path.join(paths['CHECKPOINT_PATH'])}")


# # 5. Update Config For Transfer Learning

# In[16]:


import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format


# In[17]:


config = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])


# In[18]:


config


# In[19]:


pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "r") as f:                                                                                                                                                                                                                     
    proto_str = f.read()                                                                                                                                                                                                                                          
    text_format.Merge(proto_str, pipeline_config)  


# In[20]:


pipeline_config.model.ssd.num_classes = len(labels)
pipeline_config.train_config.batch_size = 4
pipeline_config.train_config.fine_tune_checkpoint = os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-0')
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path= files['LABELMAP']
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'train.record')]
pipeline_config.eval_input_reader[0].label_map_path = files['LABELMAP']
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'test.record')]


# In[21]:


config_text = text_format.MessageToString(pipeline_config)                                                                                                                                                                                                        
with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "wb") as f:                                                                                                                                                                                                                     
    f.write(config_text) 


# # 6. Train the model

# # 8. Load Train Model From Checkpoint

# In[22]:


import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util


# In[23]:


# gpu limitation
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuratiom(
            gpus[0], [tf.config.experimental.virtualdeviceconfiguration(memory_limit=2120)])
    except RunTimeError as e:
        print(e)


# In[24]:


# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-11')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


# # 9. Detect from an Image

# In[25]:


import cv2 
import numpy as np
from matplotlib import pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[26]:


category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])


# In[27]:


IMAGE_PATH = os.path.join(paths['IMAGE_PATH'],'test2.jpg')


# In[28]:


img = cv2.imread(IMAGE_PATH)
image_np = np.array(img)

input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
detections = detect_fn(input_tensor)

num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy()
              for key, value in detections.items()}
detections['num_detections'] = num_detections

# detection_classes should be ints.
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

label_id_offset = 1
image_np_with_detections = image_np.copy()

viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes']+label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=.4,
            agnostic_mode=False)

plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
plt.show()


# # applying ocr to detection

# In[29]:


detections.keys()


# In[30]:


import easyocr


# In[31]:


detection_threshold = 0.7


# In[32]:


image = image_np_with_detections
scores = list(filter(lambda x: x>detection_threshold, detections['detection_scores']))
boxes = detections['detection_boxes'][ :len(scores)]
classes = detections['detection_classes'][ :len(scores)]


# In[33]:


width = image.shape[1]
height = image.shape[0]


# In[34]:


for idx, box in enumerate(boxes):
    print(box)
    roi = box*[height, width, height, width]
    print(roi)
    region = image[int(roi[0]):int(roi[2]),int(roi[1]):int(roi[3])]
    reader = easyocr.Reader(['en'])
    ocr_result = reader.readtext(region)
    print(ocr_result)
    plt.imshow(cv2.cvtColor(region,cv2.COLOR_BGR2RGB))


# # ocr filtering

# In[35]:


region_threshold = 0.05


# In[36]:


def filter_text(region, ocr_result,region_threshold):
    rectangle_size=region.shape[0]*region.shape[1]
    
    plate = []
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result [0][1]))
        
        if length*height / rectangle_size > region_threshold:
            plate.append(result[1])
    
    return plate  


# In[37]:


filter_text(region, ocr_result,region_threshold)


# # bring together

# In[38]:


plt.imshow(image_np_with_detections)


# In[39]:


region_threshold


# In[40]:


def ocr_it(image, detections, detection_threshold, region_threshold):
    
    #scores,boxes and classes above threshold 
    scores = list(filter(lambda x: x>detection_threshold, detections['detection_scores']))
    boxes = detections['detection_boxes'][ :len(scores)]
    classes = detections['detection_classes'][ :len(scores)]
    
    width = image.shape[1]
    height = image.shape[0]
    
    #Apply ROI filtering and ocr
    for idx, box in enumerate(boxes):
        roi = box*[height, width, height, width]
        region = image[int(roi[0]):int(roi[2]),int(roi[1]):int(roi[3])]
        reader = easyocr.Reader(['en'])
        ocr_result = reader.readtext(region)
        
        text = filter_text(region, ocr_result, region_threshold)
        
        plt.imshow(cv2.cvtColor(region,cv2.COLOR_BGR2RGB))
        plt.show()
        print(text) 
        return text, region


# In[41]:


text, region = ocr_it(image_np_with_detections, detections, detection_threshold, region_threshold)


# # saved result

# In[42]:


import csv
import uuid


# In[43]:


'{}.jpg'.format(uuid.uuid1())


# In[44]:


def save_results(text, region, csv_filename,folder_path):
    img_name = '{}.jpg'.format(uuid.uuid1())
    
    cv2.imwrite(os.path.join(folder_path, img_name),region)
    
    with open(csv_filename,mode='a', newline='') as f:
        csv_writer = csv.writer(f,delimiter = ',' ,quotechar='"', quoting = csv.QUOTE_MINIMAL)
        csv_writer.writerow([img_name, text])


# In[45]:


region


# In[46]:


save_results(text, region ,'detection_results.csv', 'detection_images')


# # 10. Real Time Detections from your Webcam

# In[47]:


get_ipython().system('pip install opencv-python-headless')


# In[49]:


cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while cap.isOpened(): 
    ret, frame = cap.read()
    image_np = np.array(frame)
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    #image_np_expanded = np.expand_dims(frame, axis=0)
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.5,
                agnostic_mode=False)
    try:
        text, region = ocr_it(image_np_with_detections, detections, detection_threshold, region_threshold)
        save_results(text, region, 'realtimeresults.csv', 'detection_images')
    except:
        pass

    cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break






