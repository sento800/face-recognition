import numpy as np
from keras_facenet import FaceNet
import faiss
import pickle
from arcface import ArcFace
import threading

def _load_pickle(file_path):
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    return obj


embed = _load_pickle('./embedded/pkl_files/arc_embeds_final.pkl')
labels = _load_pickle('./embedded/pkl_files/arc_labels_final.pkl')
# embed = _load_pickle('./file_model/pkl_files/embeds_final.pkl')
# labels = _load_pickle('./file_model/pkl_files/labels_final.pkl')

ids = np.arange(len(labels))
class Faiss:
    def __init__(self , metric : str , d : int , data , labels):
        self.metric = metric
        self.d = d
        self.data = data
        self.labels = labels
        self.index = self._create_index()
    def _create_index(self):
        if self.metric == 'euclidean':
            index = faiss.IndexFlatL2(self.d)
        elif self.metric == 'cosine':
            index = faiss.IndexFlatIP(self.d)
        else:
             raise ValueError(f"Unsupported metric: {self.metric}")
        return index
    def add_data(self):
        self.index.add(self.data)
    def predict(self , query , k :int):
        k = k
        distances, neighbors = self.index.search(query, k)
        if distances[0][0] < 0.5:
            return neighbors
        else:
            return "Unknown"

# def facenet_process_img(img):
#     embedder = FaceNet()
#     detections = embedder.extract(img)
#     embed = detections[0]['embedding']
#     return embed

# img = './data_face/dphu/image_100_dphu.jpg'
# img2 = './data_face/dphu/image_101_dphu.jpg'

def ArcFace_process_img(img):
    face_rec = ArcFace.ArcFace()
    emb1 = face_rec.calc_emb(img)
    emb1
    return emb1

def predict_face(img):
    embed_detect = np.array([ArcFace_process_img(img)])
    faiss = Faiss('euclidean'  , 512 , np.stack(embed) , labels)
    faiss.add_data()
    i = faiss.predict(embed_detect,5)
    if type(i) is str:
        predict =  i
    else:
        predict = labels[i[0][0]]
    return predict







        