import torch
import cv2
import os
import time
import numpy as np
from pygame import mixer

classes = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9}
jackpot = 9
win_bgm = "./win.mp3"
img_dir = "./win_animation"
win_animation = []
for i in range(1, len(os.listdir(img_dir))):
    print(i)
    img = cv2.imread(os.path.join(img_dir, f"{i}.jpeg"))
    win_animation.append(img)

model = model = torch.hub.load(
    "../yolov5",
    "custom",
    path="best.pt",
    source="local",
    force_reload=True,
)
model.conf = 0.7


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    mixer.init()
    mixer.music.load(win_bgm)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.flip(image, 1)
        result = model(image)

        win = False
        # if detect cards
        if len(result.xyxy[0]) > 0:
            for item in result.xyxy:
                label = classes[int(item[0][5])]
                if label == jackpot:
                    win = True
                    break

        if win:
            win = False
            counter = 0
            jackpot = np.random.randint(1, 10)
            while True:
                if not mixer.music.get_busy() and counter == 0:
                    mixer.music.play()

                cv2.imshow("Invoice Game", win_animation[counter])
                time.sleep(0.05)

                if counter < len(win_animation) - 1:
                    counter += 1

                if cv2.waitKey(5) & 0xFF == 27:
                    mixer.music.stop()
                    break

        cv2.imshow("Invoice Game", np.squeeze(result.render()))
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
