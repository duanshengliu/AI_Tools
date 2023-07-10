# -*- coding:utf-8 -*-
from absl import app, flags, logging
from tensorflow.keras.models import load_model
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.graph_util import convert_variables_to_constants

FLAGS = flags.FLAGS
flags.DEFINE_string('input_model', None, 'Path to the input model.', short_name='i')
flags.DEFINE_string('output_model', None, 'Path where the converted model will '
                                          'be stored.', short_name='o')
flags.mark_flag_as_required('input_model')
flags.mark_flag_as_required('output_model')

def main(args):
    '''
    Convert keras(.h5) to GraphDef(.pb) model file, for tensorflow 1.x / 2.x
    Reference : https://github.com/amir-abdi/keras_to_tensorflow
    '''
    input_model = FLAGS.input_model
    output_model = FLAGS.output_model

    assert input_model.endswith(".h5") or \
           input_model.endswith(".hdf5"), "input_model must endswith `.h5` or `.hdf5`"
    assert output_model.endswith(".pb"),  "output_model must endswith `.pb`"
    tf.keras.backend.clear_session()
    tf.disable_eager_execution()  # disable eager mode
    tf.keras.backend.set_learning_phase(0) # this is important
    keras_model = load_model(input_model)
    session = tf.keras.backend.get_session()
    input_names = [input.op.name for input in keras_model.inputs]
    output_names = [out.op.name for out in keras_model.outputs]
    frozen_graph_def = convert_variables_to_constants(session,
                                                      session.graph_def,
                                                      output_names)
    logging.info("Convert Successfully")
    with tf.io.gfile.GFile(output_model, "wb") as f:
        f.write(frozen_graph_def.SerializeToString())
    message = f"===== Save GraphDef Pb Model to: {output_model} =====\n"
    logging.info(message)


if __name__ == "__main__":
    app.run(main)