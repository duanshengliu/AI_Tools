# -*- coding:utf-8 -*-
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBILE_DEVICES"] = "-1"
import tensorflow.compat.v1 as tf

if tf.__version__[0] == "2":
    print("请使用tensorflow1.15环境")
    quit()
from tensorflow.core.framework import attr_value_pb2, types_pb2
from tensorflow.tools.graph_transforms import TransformGraph


_TFLite_Detection_PostProcess = "TFLite_Detection_PostProcess"
# ================= 示例参数 ======================
detection_nums = 1917
max_detections = 10
max_classes_per_detection = 1
nms_score_threshold = {9.99999993922529e-09}
nms_iou_threshold = {0.6000000238418579}
num_classes = 91
scale_values = {"y_scale": {10.0}, "x_scale": {10.0}, "h_scale": {5.0}, "w_scale": {5.0}}
detections_per_class = 100
use_regular_nms = False
additional_output_tensors = {}


# ================= 示例参数 ======================

def append_postprocessing_op(frozen_graph_def,
                             max_detections,
                             max_classes_per_detection,
                             nms_score_threshold,
                             nms_iou_threshold,
                             num_classes,
                             scale_values,
                             detections_per_class=100,
                             use_regular_nms=False,
                             additional_output_tensors=()):
    # This function is copied from https://github.com/tensorflow/models/blob/master/research/object_detection/export_tflite_ssd_graph_lib.py#L65
    """
    Appends postprocessing custom op.
    Args:
      frozen_graph_def: Frozen GraphDef for SSD model after freezing the
        checkpoint
      max_detections: Maximum number of detections (boxes) to show
      max_classes_per_detection: Number of classes to display per detection
      nms_score_threshold: Score threshold used in Non-maximal suppression in
        post-processing
      nms_iou_threshold: Intersection-over-union threshold used in Non-maximal
        suppression in post-processing
      num_classes: number of classes in SSD detector
      scale_values: scale values is a dict with following key-value pairs
        {y_scale: 10, x_scale: 10, h_scale: 5, w_scale: 5} that are used in decode
          centersize boxes
      detections_per_class: In regular NonMaxSuppression, number of anchors used
        for NonMaxSuppression per class
      use_regular_nms: Flag to set postprocessing op to use Regular NMS instead of
        Fast NMS.
      additional_output_tensors: Array of additional tensor names to output.
        Tensors are appended after postprocessing output.
    Returns:
      transformed_graph_def: Frozen GraphDef with postprocessing custom op
      appended
      TFLite_Detection_PostProcess custom op node has four outputs:
      detection_boxes: a float32 tensor of shape [1, num_boxes, 4] with box
      locations
      detection_classes: a float32 tensor of shape [1, num_boxes]
      with class indices
      detection_scores: a float32 tensor of shape [1, num_boxes]
      with class scores
      num_boxes: a float32 tensor of size 1 containing the number of detected
      boxes
    """
    new_output = frozen_graph_def.node.add()
    new_output.op = 'TFLite_Detection_PostProcess'
    new_output.name = 'TFLite_Detection_PostProcess'
    new_output.attr['_output_quantized'].CopyFrom(
        attr_value_pb2.AttrValue(b=True))
    new_output.attr['_output_types'].list.type.extend([
        types_pb2.DT_FLOAT, types_pb2.DT_FLOAT, types_pb2.DT_FLOAT,
        types_pb2.DT_FLOAT
    ])
    new_output.attr['_support_output_type_float_in_quantized_op'].CopyFrom(
        attr_value_pb2.AttrValue(b=True))
    new_output.attr['max_detections'].CopyFrom(
        attr_value_pb2.AttrValue(i=max_detections))
    new_output.attr['max_classes_per_detection'].CopyFrom(
        attr_value_pb2.AttrValue(i=max_classes_per_detection))
    new_output.attr['nms_score_threshold'].CopyFrom(
        attr_value_pb2.AttrValue(f=nms_score_threshold.pop()))
    new_output.attr['nms_iou_threshold'].CopyFrom(
        attr_value_pb2.AttrValue(f=nms_iou_threshold.pop()))
    new_output.attr['num_classes'].CopyFrom(
        attr_value_pb2.AttrValue(i=num_classes))

    new_output.attr['y_scale'].CopyFrom(
        attr_value_pb2.AttrValue(f=scale_values['y_scale'].pop()))
    new_output.attr['x_scale'].CopyFrom(
        attr_value_pb2.AttrValue(f=scale_values['x_scale'].pop()))
    new_output.attr['h_scale'].CopyFrom(
        attr_value_pb2.AttrValue(f=scale_values['h_scale'].pop()))
    new_output.attr['w_scale'].CopyFrom(
        attr_value_pb2.AttrValue(f=scale_values['w_scale'].pop()))
    new_output.attr['detections_per_class'].CopyFrom(
        attr_value_pb2.AttrValue(i=detections_per_class))
    new_output.attr['use_regular_nms'].CopyFrom(
        attr_value_pb2.AttrValue(b=use_regular_nms))

    new_output.input.extend(
        ['raw_outputs/box_encodings', 'raw_outputs/class_predictions', 'anchors'])
    # Transform the graph to append new postprocessing op
    input_names = []
    output_names = ['TFLite_Detection_PostProcess'
                    ] + list(additional_output_tensors)
    transforms = ['strip_unused_nodes']
    transformed_graph_def = TransformGraph(frozen_graph_def, input_names,
                                           output_names, transforms)
    return transformed_graph_def


def add_postprocess_op(pb_path):
    """
    1. Load the pb GraphDef(including anchor tensor) then get the session.
    2. Transform the session.graph_def by `append_postprocessing_op`
    """
    assert os.path.exists(pb_path), f"{pb_path} not exists"

    with tf.io.gfile.GFile(pb_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
        session = tf.Session(graph=graph)
    print(f"[INFO] Load Pb GraphDef model from: {pb_path} successfully")

    transformed_graph_def = append_postprocessing_op(
        session.graph_def,
        max_detections,
        max_classes_per_detection,
        nms_score_threshold,
        nms_iou_threshold,
        num_classes,
        scale_values,
        detections_per_class,
        use_regular_nms,
        additional_output_tensors
    )

    with tf.io.gfile.GFile(save_pb_path, "wb") as f:
        f.write(transformed_graph_def.SerializeToString())
    print(f"===== Save Pb GraphDef model with postprocess op to: {save_pb_path} =====\n")


def convert_pb_to_tflite(save_pb_path, tflite_path):
    """
    Convert the Pb GraphDef model with postprocess op to tflite model
    """
    print(f"[INFO] Converting {save_pb_path} to tflite ...")
    TFLiteConverter = tf.lite.TFLiteConverter
    converter = TFLiteConverter.from_frozen_graph(
        graph_def_file=save_pb_path,
        input_arrays=["normalized_input_image_tensor"],
        input_shapes={"normalized_input_image_tensor": (1, 300, 300, 3)},
        output_arrays=[_TFLite_Detection_PostProcess,
                       _TFLite_Detection_PostProcess + ":1",
                       _TFLite_Detection_PostProcess + ":2",
                       _TFLite_Detection_PostProcess + ":3",
                       ]
    )

    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.allow_custom_ops = True

    converter.inference_type = tf.uint8  # tf.lite.constants.QUANTIZED_UINT8
    input_arrays = converter.get_input_arrays()
    converter.quantized_input_stats = {input_arrays[0]: (127.5, 127.5)}  # mean, std_dev
    converter.default_ranges_stats = (0, 255)

    tflite_model = converter.convert()
    with open(tflite_path, "wb")as f:
        f.write(tflite_model)
    print(f"===== Save the TFLite model with postprocess op to: {tflite_path} =====")


if __name__ == "__main__":
    # src pb GraphDef model path
    pb_path = "ssd_mobilenet_v2.pb"
    # pb GraphDef model path with postprocessing op
    save_pb_path = pb_path.replace(".pb", "_postprocess.pb")
    # tflite model path with postprocessing op
    tflite_path = pb_path.replace(".pb", "_postprocess.tflite")

    add_postprocess_op(pb_path)

    convert_pb_to_tflite(save_pb_path, tflite_path)
