from io import BytesIO
import pickle
import cv2
import mediapipe as mp
import numpy as np


class Soz:
    def __init__(self):
        self.model_dict = pickle.load(open(r'C:\Users\PC\PROJECTS\sign-language-detector-python\model1.p', 'rb'))
        self.model = self.model_dict['model']

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.6)


        self.labels_dict = {
            -1:"Nan",
            0:"Samolyot",
            1:"Kursi",
            2:"Odam",
            3:"Davlat",
            4:"Tarix",
            5:"Ayiruv",
        }
    def frame_to_txt(self, frame):
        data_aux = []
        x_ = []
        y_ = []

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.hands.process(frame_rgb)

        pchs = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                try:
                    prediction = self.model.predict([np.asarray(data_aux)])
                except:
                    prediction = [-1]
                predicted_character = self.labels_dict[int(prediction[0])]
                pchs.append(predicted_character)
        return pchs
class Harf:
    def __init__(self):
        self.model_dict = pickle.load(open(r'C:\Users\PC\PROJECTS\sign-language-detector-python\model.p', 'rb'))
        self.model = self.model_dict['model']

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.6)


        self.labels_dict = {
            -1:"Nan",
            0:"A",
            1:"B",
            2:"E",
            3:"F",
            4:"G",
            5:"I",
            6:"J",
            7:"L",
            8:"M",
            9:"N",
            10:"O",
            11:"P",
            12:"R",
            13:"S",
            14:"T",
            15:"U",
            16:"SH",
            17:"CH",
        }

    def frame_to_txt(self, frame):
        data_aux = []
        x_ = []
        y_ = []

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.hands.process(frame_rgb)

        pchs = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                try:
                    prediction = self.model.predict([np.asarray(data_aux)])
                except:
                    prediction = [-1]
                predicted_character = self.labels_dict[int(prediction[0])]
                pchs.append(predicted_character)
        return pchs
