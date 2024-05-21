import pickle

import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open(r'C:\Users\PC\PROJECTS\sign-language-detector-python\model.p', 'rb'))
model = model_dict['model']


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.6)

seconds = {
    "A": 2,
    "B": 10,
    # "C": 18,
    # "D": 26,
    "E": 18,
    "F": 26,
    "G": 34,
    # "H": 8,
    "I": 43,
    "J": 55,
    # "K": ,
    "L": 64,
    "M": 73,
    "N": 82,
    "O": 91,
    "P": 101,
    # "Q": ,
    "R": 111,
    "S": 122,
    "T": 132,
    "U": 143,
    # "V": 22,
    # "X": 23,
    # "Y": 24,
    # "Z": 25,
    # "NG": 26,
    "SH": 152,
    "CH": 165,
}


labels_dict = {
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

# for i in labels_dict:
#     try:
#         cap = cv2.VideoCapture(rf"C:\Users\PC\PycharmProjects\malumotlar_tuzilmasi_va_algoritmlash\vide_cat\{i}_{labels_dict[i]}.mp4")
#         while True:
#
#             data_aux = []
#             x_ = []
#             y_ = []
#
#             ret, frame = cap.read()
#
#             H, W, _ = frame.shape
#
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#             results = hands.process(frame_rgb)
#             if results.multi_hand_landmarks:
#                 for hand_landmarks in results.multi_hand_landmarks:
#                     mp_drawing.draw_landmarks(
#                         frame,  # image to draw
#                         hand_landmarks,  # model output
#                         mp_hands.HAND_CONNECTIONS,  # hand connections
#                         mp_drawing_styles.get_default_hand_landmarks_style(),
#                         mp_drawing_styles.get_default_hand_connections_style())
#
#                 for hand_landmarks in results.multi_hand_landmarks:
#                     for i in range(len(hand_landmarks.landmark)):
#                         x = hand_landmarks.landmark[i].x
#                         y = hand_landmarks.landmark[i].y
#
#                         x_.append(x)
#                         y_.append(y)
#
#                     for i in range(len(hand_landmarks.landmark)):
#                         x = hand_landmarks.landmark[i].x
#                         y = hand_landmarks.landmark[i].y
#                         data_aux.append(x - min(x_))
#                         data_aux.append(y - min(y_))
#
#                 x1 = int(min(x_) * W) - 10
#                 y1 = int(min(y_) * H) - 10
#
#                 x2 = int(max(x_) * W) - 10
#                 y2 = int(max(y_) * H) - 10
#
#                 try:
#                     prediction = model.predict([np.asarray(data_aux)])
#                 except:
#                     prediction = [-1]
#                 predicted_character = labels_dict[int(prediction[0])]
#
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
#                 cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
#                             cv2.LINE_AA)
#
#             cv2.imshow('frame', frame)
#             cv2.waitKey(1)
#
#
#         cap.release()
#         cv2.destroyAllWindows()
#     except:
#         pass

# cap = cv2.VideoCapture(rf"C:\Users\PC\Favorites\Downloads\Telegram Desktop\video_2024-03-16_10-14-39.mp4")
cap = cv2.VideoCapture(0)
while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

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
                prediction = model.predict([np.asarray(data_aux)])
                print(prediction)
            except:
                prediction = [-1]
            predicted_character = labels_dict[int(prediction[0])]
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10
            x_, y_, data_aux = [], [], []
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

        # x1 = int(min(x_) * W) - 10
        # y1 = int(min(y_) * H) - 10
        #
        # x2 = int(max(x_) * W) - 10
        # y2 = int(max(y_) * H) - 10

        # try:
        #     prediction = model.predict([np.asarray(data_aux)])
        # except:
        #     prediction = [-1]
        # predicted_character = labels_dict[int(prediction[0])]
        #
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        # cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
        #             cv2.LINE_AA)

    cv2.imshow('Harflar', frame)
    cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()
# cap = cv2.VideoCapture(rf"C:\Users\PC\Favorites\Downloads\Telegram Desktop\video_2024-03-16_10-14-39.mp4")
# while True:
#
#     data_aux = []
#     x_ = []
#     y_ = []
#
#     ret, frame = cap.read()
#
#     H, W, _ = frame.shape
#
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     results = hands.process(frame_rgb)
#
#     print(results.multi_hand_landmarks)
#
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 frame,  # image to draw
#                 hand_landmarks,  # model output
#                 mp_hands.HAND_CONNECTIONS,  # hand connections
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style())
#
#         for hand_landmarks in results.multi_hand_landmarks:
#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y
#
#                 x_.append(x)
#                 y_.append(y)
#
#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y
#                 data_aux.append(x - min(x_))
#                 data_aux.append(y - min(y_))
#
#         x1 = int(min(x_) * W) - 10
#         y1 = int(min(y_) * H) - 10
#
#         x2 = int(max(x_) * W) - 10
#         y2 = int(max(y_) * H) - 10
#
#         try:
#             prediction = model.predict([np.asarray(data_aux)])
#         except:
#             prediction = [-1]
#         predicted_character = labels_dict[int(prediction[0])]
#
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
#         cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
#                     cv2.LINE_AA)
#
#     cv2.imshow('frame', frame)
#     cv2.waitKey(1)
#
#
# cap.release()
# cv2.destroyAllWindows()
