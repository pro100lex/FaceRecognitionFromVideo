import pickle
import face_recognition
from cv2 import cv2


def detected_person_from_video(path_encodings, path_video):
    """Функция распознавания лица на видео"""
    normalized_path_encodings = path_encodings.replace('"', '').replace('\\', '/')
    normalized_path_video = path_video.replace('"', '').replace('\\', '/')

    data = pickle.loads(open(normalized_path_encodings, 'rb').read())
    video_cap = cv2.VideoCapture(normalized_path_video)

    while True:
        is_correct, frame = video_cap.read()

        face_coordinates = face_recognition.face_locations(frame, model='cnn')
        face_encodings = face_recognition.face_encodings(frame, face_coordinates)

        for face_encoding, face_coordinate in zip(face_encodings, face_coordinates):
            result_compare = face_recognition.compare_faces(data['encodings_list'], face_encoding)
            face_match = None

            if True in result_compare:
                face_match = data['name']
                print(f'[ИНФОРМАЦИЯ] Найдено совпадение: {face_match}!')
            else:
                print('[ИНФОРМАЦИЯ] Совпадений не найдено!')

            face_left_top_border = (face_coordinate[3], face_coordinate[0])
            face_right_bottom_border = (face_coordinate[1], face_coordinate[2])
            color_border = [0, 255, 0]
            cv2.rectangle(frame, face_left_top_border, face_right_bottom_border, color_border, 4)

            face_left_bottom_border_text = (face_coordinate[3], face_coordinate[2])
            face_right_bottom_border_text = (face_coordinate[1], face_coordinate[2] + 20)
            cv2.rectangle(frame, face_left_bottom_border_text, face_right_bottom_border_text, color_border, cv2.FILLED)
            cv2.putText(frame, face_match, (face_coordinate[3] + 10, face_coordinate[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)

        cv2.imshow('Программа распознавания лиц на видео', frame)

        key = cv2.waitKey(20)

        if key == ord('q'):
            print('Нажата клавиша Q, завершение процесса...')
            break


def main():
    path_encodings = input('Путь до файла с моделью: ')
    path_video = input('Путь до видео: ')
    print('Для завершения процесса нажмите клавишу Q (Включить английскую раскладку)')
    detected_person_from_video(path_encodings=path_encodings, path_video=path_video)


if __name__ == '__main__':
    main()