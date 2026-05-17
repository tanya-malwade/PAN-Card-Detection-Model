# import streamlit as st
# from ultralytics import YOLO
# from PIL import Image
# import numpy as np

# st.set_page_config(page_title="PAN Card Detector", layout="wide")

# st.title("💳 PAN Card Detection System")

# # load model
# model = YOLO("runs/detect/train/weights/best.pt")

# uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

# if uploaded_file is not None:

#     image = Image.open(uploaded_file)

#     col1, col2 = st.columns(2)

#     with col1:
#         st.image(image, caption="Uploaded Image", use_container_width=True)

#     if st.button("Detect Card"):

#         img_np = np.array(image)

#         results = model(img_np)

#         result = results[0]

#         detected_classes = result.boxes.cls.tolist()

#         names = model.names

#         labels = [names[int(i)] for i in detected_classes]

#         result_img = result.plot()

#         with col2:
#             st.image(result_img, caption="Detection Result", use_container_width=True)

#         st.subheader("Detection Result")

#         if len(labels) == 0:
#             st.error("No card detected")

#         else:
#             for label in labels:
#                 if "pancard" in labels:
#                     st.success("💳 PAN Card Detected ✅")
#                 else:
#                     st.warning("⚠ This is not a PAN Card")

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

st.set_page_config(page_title="PAN Card Detection", layout="wide")

st.title("💳 PAN Card Detection System")

# load YOLO model
model = YOLO("runs/detect/train/weights/best.pt")

# load PAN template
template = cv2.imread("PanCard_template.jpeg", 0)

uploaded_file = st.file_uploader("Upload Card Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    img_np = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Detect Card"):

        results = model(img_np)
        result = results[0]

        result_img = result.plot()

        with col2:
            st.image(result_img, caption="Detected Image", use_container_width=True)

        if len(result.boxes) == 0:
            st.error("No card detected")

        else:

            # crop detected card
            box = result.boxes.xyxy[0].cpu().numpy().astype(int)
            x1,y1,x2,y2 = box
            card = img_np[y1:y2, x1:x2]

            card_gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)

            # ORB feature detector
            orb = cv2.ORB_create()

            kp1, des1 = orb.detectAndCompute(template, None)
            kp2, des2 = orb.detectAndCompute(card_gray, None)

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            matches = bf.match(des1, des2)

            matches = sorted(matches, key=lambda x: x.distance)

            similarity = len(matches)

            st.subheader("Similarity Score")
            st.write(similarity)

            if similarity > 40:
                st.success("💳 PAN Card Detected ✅")
            else:
                st.error("❌ Not a PAN Card")

