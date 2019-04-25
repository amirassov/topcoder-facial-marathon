import cv2
import numpy as np
import mxnet as mx
from sklearn.preprocessing import normalize

from reid.insightface.mtcnn import MtcnnDetector
from reid.insightface.utils import preprocess


def get_embedder(ctx, image_size, model_prefix: str, layer):
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, 0)
    all_layers = sym.get_internals()
    sym = all_layers[layer + '_output']
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    return model


class ArcFaceModel:
    def __init__(self, embedder_path, mtcnn_path, image_size=(112, 112)):
        self.image_size = image_size
        self.ctx = mx.cpu()
        self.embedder = get_embedder(self.ctx, image_size, embedder_path, 'fc1')
        self.detector = MtcnnDetector(
            model_folder=mtcnn_path,
            ctx=self.ctx,
            accurate_landmark=True,
            threshold=[0.6, 0.7, 0.8]
        )

    def predict(self, image):
        embedding = None
        preprocessed_img, bbox, landmark = self.detect(image)
        if preprocessed_img is not None:
            embedding = self.embed(preprocessed_img)
        return embedding

    def align(self, image, bbox, landmark):
        landmark = landmark.reshape((2, 5)).T
        preprocessed_img = preprocess(image, bbox, landmark, image_size=self.image_size)
        preprocessed_img = cv2.cvtColor(preprocessed_img, cv2.COLOR_BGR2RGB)
        preprocessed_img = np.transpose(preprocessed_img, (2, 0, 1))
        return preprocessed_img, bbox, landmark

    def detect(self, image):
        bboxes, landmarks = self.detector.detect_face(image)
        if bboxes is None:
            return None, None, None

        bboxes, scores = bboxes[:, :4], bboxes[:, 4]
        return self.align(image, bboxes[0], landmarks[0])

    def embed(self, image):
        input_blob = np.expand_dims(image, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.embedder.forward(db, is_train=False)
        embedding = self.embedder.get_outputs()[0].asnumpy()
        embedding = normalize(embedding).flatten()
        return embedding
