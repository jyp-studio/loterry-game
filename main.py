import torch
import cv2
import os
import time
import numpy as np
from pygame import mixer

print("load settings")
classes = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9}
jackpot = 9
max_loss_times = 10  # only hit the jackpot after loss many times.
win_bgm = "./win.mp3"
lose_bgm = "./lose.mp3"
img_dir = "./win_animation"
win_animation = []
for i in range(1, len(os.listdir(img_dir))):
    img = cv2.imread(os.path.join(img_dir, f"{i}.jpeg"))
    win_animation.append(img)
print("load settings done")
print(f"There are {len(win_animation)} images in win_animation.")

print("load yolov5 model")
model = model = torch.hub.load(
    "./yolov5",
    "custom",
    path="best.pt",
    source="local",
    force_reload=True,
)
model.conf = 0.7
print("load model done")
print("model confident:", model.conf)

if __name__ == "__main__":
    print("open camera")
    cap = cv2.VideoCapture(0)

    print("load music")
    mixer.init()
    win_soundeffect = mixer.Sound(win_bgm)
    lose_soundeffect = mixer.Sound(lose_bgm)

    start_loss_time = 0
    loss_times = 0

    print("start game loop")
    print("====================================")
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.flip(image, 1)
        result = model(image)
        frame = np.squeeze(result.render())

        win = False
        lose = False
        # if detect cards
        if len(result.xyxy[0]) > 0:
            for item in result.xyxy:
                label = classes[int(item[0][5])]
                if label == jackpot:
                    if loss_times > max_loss_times:
                        win = True
                        loss_times = 0
                        break
                    else:
                        if time.time() - start_loss_time > 1.5:
                            lose = True
                            loss_times += 1
                            start_loss_time = time.time()
                            break
                if label != jackpot and time.time() - start_loss_time > 1.5:
                    lose = True
                    loss_times += 1
                    start_loss_time = time.time()
                    break
        print(loss_times)
        # prevent win & lose events from being triggered simultaneously
        if win and lose:
            lose = False

        if win:
            lose_soundeffect.stop()
            win = False
            counter = 0
            jackpot = np.random.randint(1, 10)
            print("New jackpot:", jackpot)
            while True:
                if not mixer.music.get_busy() and counter == 0:
                    win_soundeffect.play()

                cv2.imshow("Invoice Game", win_animation[counter])
                time.sleep(0.05)

                if counter < len(win_animation) - 1:
                    counter += 1

                if cv2.waitKey(5) & 0xFF == 27:
                    win_soundeffect.stop()
                    break

        if lose:
            win_soundeffect.stop()
            lose = False
            if not mixer.music.get_busy():
                lose_soundeffect.play()

        cv2.imshow("Invoice Game", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
