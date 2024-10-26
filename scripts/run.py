import cv2
import numpy as np
import tensorflow as tf

print(tf.__file__)  # Should output 2.8.0

from tensorflow.keras.models import model_from_json
import pyautogui  # For simulating keypress


if __name__ == "__main__":

    # Load the pre-trained model
    json_file = open('final_model_3dcnn_lstm.json', 'r')
    loaded_json_model = json_file.read()
    json_file.close()
    model = model_from_json(loaded_json_model)
    model.load_weights("final_model_3dcnn_lstm.h5")

    print("Model loaded successfully")

    # Start the camera
    camera = cv2.VideoCapture(0)

    # Load the gesture classes
    #classes = np.load("encoder_classes.npy", allow_pickle=True).tolist()[0]
    classes=['No gesture', 'Swiping Left', 'Swiping Right', 'Thumb Down', 'Thumb Up', 'Zooming In With Full Hand', 'Zooming Out With Full Hand']
    #classes=['Swiping Right','Swiping Left','Thumb Up','Thumb Down','No gesture','Zooming In With Full Hand','Zooming Out With Full Hand','Doing other things']
    #classes = ['No gesture','Swiping Left','Swiping Right','Zooming In With Full Hand','Zooming Out With Full Hand']
    #classes=['No gesture', 'Swiping Left', 'Swiping Right', 'Thumb Down', 'Thumb Up', 'Zooming In With Full Hand', 'Zooming Out With Full Hand']
    print("Classes:", classes)

    num_frames = 0
    li = []
    top3 = [0, 1, 2]
    output = [0, 0, 0, 0, 0, 0, 0, 0]
    label = "no action"
    cooldown = 0
    dic = {"Swiping Left": 'right', "Swiping Right": 'left',"Zooming In With Full Hand": 'esc', "Zooming Out With Full Hand": 'esc' }
    key_pressed = None  # To track the currently pressed key

    while True:
        (grabbed, frame) = camera.read()
        frame = cv2.flip(frame, 1)

        (height, width) = frame.shape[:2]
        inp = cv2.resize(frame, (32, 32))

        if num_frames < 36:
            li.append(inp)

        if num_frames >= 36:
            li.pop(0)
            li.append(inp)

        # Every 18 frames (after the first 36 frames), make a prediction
        if num_frames > 35 and (num_frames - 36) % 18 == 0 and cooldown == 0:
            inp_video = np.asarray(li)[None, :]
            output = model.predict(inp_video, verbose=1)[0][1:]
            #print(output)
            index = np.argmax(output)
            #print(index)
            top3 = output.argsort()[-3:]  # Top 3 predictions
            #print(top3)
            label = classes[index]
            if label == "Doing other things":
                label = "No gesture"

            print("Predicted gesture:", label)

            # Perform action if Swiping Left or Swiping Right is detected
            if label in dic:
                if key_pressed is not dic[label]:  # Only press if it's a different key
                    if key_pressed:  # Release the previously pressed key
                        pyautogui.keyUp(key_pressed)
                        print(f"Released {key_pressed} key")
                    pyautogui.keyDown(dic[label])  # Press the new key
                    key_pressed = dic[label]
                    print(f"Pressed {dic[label]} key")

            else:
                # No recognized gesture, release any currently pressed key
                if key_pressed:
                    pyautogui.keyUp(key_pressed)
                    print(f"Released {key_pressed} key")
                    key_pressed = None

            # Cooldown period to prevent multiple triggers
            if index > 0:
                cooldown = 15 #was earlier 32, reduced it to prevent too much lag

        if cooldown > 0:
            cooldown -= 1

        num_frames += 1

        # Display the activity on the preview window
        text = f"Activity: {label}"
        backg = np.full((480, 1200, 3), 15, np.uint8)
        backg[:480, :640] = frame

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(backg, text, (40, 40), font, 1, (0, 0, 0), 2)
        for i, top in enumerate(top3):
            cv2.putText(backg, classes[top], (700, 200 - 70 * i), font, 1, (0, 255, 0), 1)
            cv2.rectangle(backg, (700, 225 - 70 * i), (int(700 + output[top] * 170), 205 - 70 * i), (255, 255, 255), 3)

        cv2.imshow('preview', backg)

        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord("q"):  # Press 'q' to quit
            break

    # Ensuring any pressed key is released before exiting
    if key_pressed:
        pyautogui.keyUp(key_pressed)
        print(f"Released {key_pressed} key")

    camera.release()
    cv2.destroyAllWindows()
