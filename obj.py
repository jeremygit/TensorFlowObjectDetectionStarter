import os
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

PATH_WORKSPACE = 'workspace'
PATH_SCRIPTS = PATH_WORKSPACE + '/scripts'
PATH_IMAGES = PATH_WORKSPACE + '/images'
PATH_MODELS = PATH_WORKSPACE + '/models' # trained model will go here
PATH_CONFIG = PATH_MODELS + '/net/pipeline.config' # network details
PATH_CHECKPOINT = PATH_MODELS + '/net' 
PATH_ANNOTATIONS = PATH_WORKSPACE + '/annotations' #annotation label_map.pbtext and tf records
PATH_PRETRAINED_MODELS = PATH_WORKSPACE + '/pre-trained-models' # pretrained models for transfer learning, from?
PATH_API_MODEL = 'Tensorflow/models' #tf object detect library model

# Label Map e.g. ID labels
# Update with own labels
labels = [
  {'name': '<label1_name>', 'id': 1},
  {'name': '<label2_name>', 'id': 2},
  {'name': '<label3_name>', 'id': 3},
  {'name': '<label4_name>', 'id': 4},
  {'name': '<label5_name>', 'id': 5}
]

with open(PATH_ANNOTATIONS + '/label_map.pbtext', 'w+') as f:
  for label in labels:
    f.write('item{\n')
    f.write('\tname:\'' + str(label['name']) + '\'\n')
    f.write('\tid:' + str(label['id']) + '\n')
    f.write('}\n')

os.system('python ' + PATH_SCRIPTS + '/generate_tfrecord.py -x ' + PATH_IMAGES + '/training -l ' + PATH_ANNOTATIONS + '/label_map.pbtext -o ' + PATH_ANNOTATIONS + '/train.record')
os.system('python ' + PATH_SCRIPTS + '/generate_tfrecord.py -x ' + PATH_IMAGES + '/test -l ' + PATH_ANNOTATIONS + '/label_map.pbtext -o ' + PATH_ANNOTATIONS + '/test.record')

PATH_PIPELINE_CONFIG = PATH_CHECKPOINT + '/pipeline.config'
config = config_util.get_configs_from_pipeline_file(PATH_PIPELINE_CONFIG)
pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()

with tf.io.gfile.GFile(PATH_PIPELINE_CONFIG, 'r') as f:
  proto_str = f.read()
  text_format.Merge(proto_str, pipeline_config)

pipeline_config.model.ssd.num_classes = len(labels)
pipeline_config.train_config.batch_size = 4
pipeline_config.train_config.fine_tune_checkpoint = PATH_PRETRAINED_MODELS + '/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/checkpoint/ckpt-0'
pipeline_config.train_config.fine_tune_checkpoint_type = 'detection' # because of the type of model
pipeline_config.train_input_reader.label_map_path = PATH_ANNOTATIONS + '/label_map.pbtext'
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [PATH_ANNOTATIONS + '/train.record']
pipeline_config.eval_input_reader[0].label_map_path = PATH_ANNOTATIONS + '/label_map.pbtext'
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [PATH_ANNOTATIONS + '/test.record']

config_text = text_format.MessageToString(pipeline_config)
with tf.io.gfile.GFile(PATH_CONFIG, 'wb') as f:
  f.write(config_text)

print('--- Training Start ---')
print('python ' + PATH_API_MODEL + '/research/object_detection/model_main_tf2.py --model_dir ' + PATH_CHECKPOINT + ' --pipeline_config_path ' + PATH_CONFIG + ' --num_train_steps=5000');
os.system('python ' + PATH_API_MODEL + '/research/object_detection/model_main_tf2.py --model_dir ' + PATH_CHECKPOINT + ' --pipeline_config_path ' + PATH_CONFIG + ' --num_train_steps=5000')