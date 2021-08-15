import cv2
import os
from random import shuffle


DATA_FOLDER_PATH = "data"
MIN_CONTOUR_WIDTH = 30
MIN_CONTOUR_HEIGHT = 30
BUFFER_SIZE = 10
VAL_SIZE = 0.02

if not os.path.exists(f"{DATA_FOLDER_PATH}/train"):
    os.mkdir(f"{DATA_FOLDER_PATH}/train")
if not os.path.exists(f"{DATA_FOLDER_PATH}/val"):
    os.mkdir(f"{DATA_FOLDER_PATH}/val")

files = os.listdir(DATA_FOLDER_PATH)
shuffle(files)
print(f"# files: {len(files)}")
val_files = 0
train_files = 0

for index, image_path in enumerate(files):
    if image_path.endswith(("png", "jpg", "jpeg")):

        image = cv2.imread(os.path.join(DATA_FOLDER_PATH, image_path))

        # convert the image to grayscale format
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # apply binary thresholding
        ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)

        # find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # blank (white) image to draw contours
        image_copy = image.copy()
        image_copy[:, :, :] = 255.

        for idx, c in enumerate(contours):
            # get a bounding box from contour points
            x, y, w, h = cv2.boundingRect(c)
            # use only large enough contours
            if w > MIN_CONTOUR_WIDTH or h > MIN_CONTOUR_HEIGHT:
                cv2.drawContours(image_copy, [c], -1, (0, 0, 0), 1, cv2.LINE_AA)
                # cv2.rectangle(image_copy,(x,y),(x+w,y+h),(155,155,0),1)

        name = image_path.replace(".png", "_contour.png")

        if index < len(files) * VAL_SIZE:
            os.rename(os.path.join(DATA_FOLDER_PATH, image_path), os.path.join(DATA_FOLDER_PATH, "val", image_path))
            cv2.imwrite(os.path.join(DATA_FOLDER_PATH, "val", name), image_copy)
            val_files += 1
        else:
            os.rename(os.path.join(DATA_FOLDER_PATH, image_path), os.path.join(DATA_FOLDER_PATH, "train", image_path))
            cv2.imwrite(os.path.join(DATA_FOLDER_PATH, "train", name), image_copy)
            train_files += 1

print(f"# train files: {train_files}")
print(f"# val files: {val_files}")

# cv2.imshow('EXTERNAL', image_copy)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


