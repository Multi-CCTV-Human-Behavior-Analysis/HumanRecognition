import numpy as np
import cv2
import tensorflow as tf

class ImageEncoder:
    def __init__(self, model_filename, input_name='images', output_name='features'):
        self.graph = tf.Graph()
        with tf.io.gfile.GFile(model_filename, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        with self.graph.as_default():
            tf.import_graph_def(graph_def, name="net")
        self.session = tf.compat.v1.Session(graph=self.graph)
        self.input_var = self.graph.get_tensor_by_name("net/" + input_name + ":0")
        self.output_var = self.graph.get_tensor_by_name("net/" + output_name + ":0")

    def __call__(self, image_batch):
        return self.session.run(self.output_var, feed_dict={self.input_var: image_batch})

def extract_image_patch(image, bbox, patch_shape=(64, 128)):
    x, y, w, h = bbox
    image_h, image_w = image.shape[:2]
    x1 = max(int(x), 0)
    y1 = max(int(y), 0)
    x2 = min(int(x + w), image_w - 1)
    y2 = min(int(y + h), image_h - 1)
    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        return np.random.randint(0, 255, (patch_shape[1], patch_shape[0], 3)).astype(np.uint8)
    return cv2.resize(cropped, (patch_shape[0], patch_shape[1]))  # ✅ width, height

def create_box_encoder(model_filename, batch_size=32):
    image_encoder = ImageEncoder(model_filename)
    def encoder(image, boxes):
        image_patches = [extract_image_patch(image, box) for box in boxes]
        image_batch = np.array(image_patches).astype(np.uint8)
        return image_encoder(image_batch)
    return encoder
