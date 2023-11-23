# bboxes.py
# ver 2.1. упорядочивание прямоугольников по их расположению на оси Х
import numpy as np
import cv2

# Функция для проверки условия: ширина больше высоты (нет ли большого поворота счетчика)
def check_rotation(input_box):
    # Вычисление ширины и высоты bounding box
    width = input_box[2] - input_box[0]  # x2 - x1
    height = input_box[3] - input_box[1] # y2 - y1

    # Проверка условия: ширина больше высоты
    if width > height:
        wh_check = "✅ W > H"
    else:
        wh_check = "❌ W ≤ H"

    return wh_check


def digits_output(class_id, input_box, image):
    # Расчет размера текста
    text = f"{class_id}" if 0 <= class_id <= 9 else ""
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    text_width, text_height = text_size

    # Координаты для темного фона
    bg_top_left = (int(input_box[0]) + 1, int(input_box[3]) + 4)
    bg_bottom_right = (int(input_box[0]) + text_width, int(input_box[3]) + text_height + 14)

    # Рисование темного прямоугольника
    cv2.rectangle(image, bg_top_left, bg_bottom_right, (28, 28, 28), cv2.FILLED)  # (128, 128, 128)

    # Вывод текста поверх серого фона
    text_position = (int(input_box[0]), int(input_box[3]) + text_height + 10)
    contrast_color = (255, 255, 255)  # Белый для контраста
    cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, contrast_color, 2)


def draw_boxes(image: np.ndarray, results: list) -> np.ndarray:
    wh_check = ""
    detected_classes = ""  # Переменная для хранения последовательности классов
    class_positions = []  # Список для хранения классов и их позиций
    number = 0
    speed = None
    # Определение цветов для классов от 0 до 9
    colors = [(255, 0, 0), (0, 205, 0), (10, 10, 205), (255, 255, 0), 
              (255, 0, 255), (10, 105, 10), (128, 0, 0), (0, 128, 0), 
              (0, 0, 128), (128, 128, 0)]

    for result in results:
        boxes = result.boxes
        speed = round(result.speed['inference'], 1)
        for bbox, score, cl in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
            class_id = int(cl)
            number += 1
            score_label = f"{score:.2f}"
            input_box = np.array(bbox)  # Координаты левого верхнего угла и ширина и высота бокса
            vertex = (int(input_box[0]), int(input_box[1]))  # Координаты вершины бокса
            left = 0
            down = -10

            # Добавление класса в последовательность
            if 0 <= class_id <= 9:
                class_positions.append((class_id, bbox[0]))  # bbox[0] это X координата левого верхнего угла
                # detected_classes += str(class_id)
                color = colors[class_id]  # Уникальный цвет для класса
                # class_label = f"{class_id}"  # Класс
                digits_output(class_id, input_box, image)
                thickness = 2
            elif class_id == 10:
                wh_check = check_rotation(input_box)
                color = (30, 255, 30)  # Уникальный цвет для рамки "основное"
                cv2.circle(image, vertex, radius=3, color=(30, 255, 30), thickness=-1) #  точка
                # class_label = f"{class_id}"  # Класс
                thickness = 1
                left = 45
                down = 12
            elif class_id == 12:
                color = (35, 235, 0)  # Уникальный цвет для рамки "номер"
                cv2.circle(image, vertex, radius=4, color=(35, 235, 0), thickness=-1) # точка
                thickness = 2
                left = 45
                down = 12
            else:
                color = (10, 180, 10)  # Общий цвет для всех остальных классов
                # class_label = ""
                thickness = 1

            cv2.rectangle(image, (int(input_box[0]), int(input_box[1])), (int(input_box[2]), int(input_box[3])), color, thickness)
            # Вывод вероятности сверху рамки
            cv2.putText(image, score_label, (int(input_box[0])-left, int(input_box[1])+down), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        

    # Сортировка классов по X координате
    class_positions.sort(key=lambda x: x[1])

    # Добавление отсортированных классов в detected_classes
    for class_id, _ in class_positions:
        detected_classes += str(class_id)

    # Вывод последовательности классов
    print(f"\n🔘 Detected digits [{wh_check}]: {detected_classes} [{number}] [{speed}]")
    return image, detected_classes, number, speed, wh_check

