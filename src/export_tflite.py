import torch
import argparse
import onnx, os
import tensorflow as tf
from onnx_tf.backend import prepare
from .model_crnn import LitCRNN
from .config import TrainConfig

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', required=True)
parser.add_argument('--out_dir', default='artifacts/export')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
cfg = TrainConfig()
model = LitCRNN.load_from_checkpoint(args.ckpt, cfg=cfg).model.eval()

# 1) Torch -> ONNX
dummy = torch.randn(1,1,300,cfg.n_mels)  # ~3s segment
onnx_path = os.path.join(args.out_dir, 'crnn.onnx')
torch.onnx.export(model, dummy, onnx_path, input_names=['x'], output_names=['logits','exac'],
                  dynamic_axes={'x':{0:'B',2:'T'}, 'logits':{0:'B'}, 'exac':{0:'B',1:'T'}},
                  opset_version=17)

# 2) ONNX -> TF (SavedModel)
tf_rep = prepare(onnx.load(onnx_path))
tf_path = os.path.join(args.out_dir, 'crnn_tf')
tf_rep.export_graph(tf_path)

# 3) TF -> TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite = converter.convert()
tflite_path = os.path.join(args.out_dir, 'crnn.tflite')
open(tflite_path, 'wb').write(tflite)
print("Exported:", tflite_path)
