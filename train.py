import os
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
import argparse

# Parser
parser = argparse.ArgumentParser(description='Configure training...')

parser.add_argument('-a', '--annotations', type=str, default='annotations', 
                    help='Specify different path for tfrecords')
parser.add_argument('-c', '--pretrained_ckpt', type=str, default='pre-trained-models/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint/ckpt-0', 
                    help='Specify path to tf zoo model ckpt 0 file')
parser.add_argument('-p', '--pipeline', type=str, default='models/ssd_mobnet_320/pipeline.config', 
                    help='Specify different path for pipeline.config file')
parser.add_argument('-o', '--object_det_api', type=str, default='API/models', 
                    help='Specify path for Object detection API directory')
parser.add_argument('-i', '--images', type=str, default='images', 
                    help='Specify Path for dataset images')

args = parser.parse_args()


# Setting up paths
SCRIPTS_PATH = 'scripts'
APIMODEL_PATH = args.object_det_api
ANNOTATION_PATH = args.annotations
IMAGE_PATH = args.images
MODEL_PATH = 'models'
PRETRAINED_MODEL_PATH = 'pre-trained-models'
CONFIG_PATH = MODEL_PATH+'/ssd_mobnet_320/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH+'/ssd_mobnet_320/'

# Label Map
labels = [{'name':'single', 'id':1}, {'name':'double', 'id':2}, {'name':'pinch', 'id':3}]

with open(ANNOTATION_PATH + '/label_map.pbtxt', 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')

# Create tf Records from XMLs
os.system("python {}'/generate_tfrecord.py' -x {}'/train' -l {}'/label_map.pbtxt' -o {}'/train.record'".format(SCRIPTS_PATH, IMAGE_PATH, ANNOTATION_PATH, ANNOTATION_PATH))
os.system("python {}'/generate_tfrecord.py' -x {}'/test' -l {}'/label_map.pbtxt' -o {}'/test.record'".format(SCRIPTS_PATH, IMAGE_PATH, ANNOTATION_PATH, ANNOTATION_PATH))

CUSTOM_MODEL_NAME = 'ssd_mobnet_320' 
os.system("mkdir 'models/'{}".format(CUSTOM_MODEL_NAME))
os.system("cp {}'/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/pipeline.config' {}'/'{}".format(PRETRAINED_MODEL_PATH, MODEL_PATH, CUSTOM_MODEL_NAME))

config = config_util.get_configs_from_pipeline_file(CONFIG_PATH)

pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(CONFIG_PATH, "r") as f:                                                                                                                                                                                                                     
    proto_str = f.read()                                                                                                                                                                                                                                       
    text_format.Merge(proto_str, pipeline_config)  

pipeline_config.model.ssd.num_classes = 3
pipeline_config.train_config.batch_size = 4
pipeline_config.train_config.fine_tune_checkpoint = PRETRAINED_MODEL_PATH+args.pretrained_ckpt
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path= ANNOTATION_PATH + '/label_map.pbtxt'
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [ANNOTATION_PATH + '/train.record']
pipeline_config.eval_input_reader[0].label_map_path = ANNOTATION_PATH + '/label_map.pbtxt'
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [ANNOTATION_PATH + '/test.record']

config_text = text_format.MessageToString(pipeline_config)                                                                                                                                                                                                        
with tf.io.gfile.GFile(CONFIG_PATH, "wb") as f:                                                                                                                                                                                                                     
    f.write(config_text)   

print("""python {}/research/object_detection/model_main_tf2.py --model_dir={}/{} --pipeline_config_path={}/{}/pipeline.config --num_train_steps=5000""".format(APIMODEL_PATH, MODEL_PATH,CUSTOM_MODEL_NAME,MODEL_PATH,CUSTOM_MODEL_NAME))
print("\n Copy the above command and run to start training")