import numpy as np
from sklearn.model_selection import train_test_split
from keras_facenet import FaceNet
import faiss
import pickle

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
            return "Không có trong dữ liệu."

class facenet_process_img:
    def __init__(self,img):
        self.img = img
    def covert(self):
        embedder = FaceNet()
        detections = embedder.extract(img, threshold=0.95)
        embed = np.array(detections[0]['embedding'])
        return embed

img = './data_face/phu/image_100_dphu.jpg'
def predict_face():
    def _load_pickle(file_path):
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
        return obj

    embed = _load_pickle('./file_model/pkl_files/embeds_test_2.pkl')
    labels = _load_pickle('./file_model/pkl_files/labels_test_2.pkl')

    x = embed

    ids = np.arange(len(labels))
    X_train, x_text, y_train, y_text, id_train, id_test = train_test_split(np.stack(embed), labels, ids, test_size = 0.2,stratify = labels , random_state=42)

    embed = np.array([facenet_process_img.covert (img)])
    faiss = Faiss('euclidean'  , X_train.shape[1] , X_train , y_train)
    faiss.add_data()
    i = faiss.predict(embed,5)

    return y_train[i[0][0]]


        