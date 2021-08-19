import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import cv2
import random
import glob

st.set_page_config(layout="wide")

# st.title('U-2 net & Swin Tranformer on Ghetty Images')

"""
# Human Pose Estimation
## Comparing Performance of OpenPose & AlphaPose
on Leeds DataSet

"""

st.markdown('##')



df = pd.DataFrame(
    [["OpenPose (CMU-Pose)", "61.8", "84.9", "67.5", "136 MB"],["AlphaPose", "73.3", "89.2", "79.1", "209 MB"]],
    columns=(['model tested', 'AP@0.5:0.95', 'mAP@0.5', 'AP@0.75', 'model size']))


# df = pd.DataFrame(
#     [["Full size U-2 Net", "salient object detection", "176.3 MB", "DUTS-TR", ""], ["Lightweight U-2 Net ", "salient object detection", "4.7 MB", "DUTS-TR", ""],["Mask R-CCN w/ Swin-T Backbone", "object detection + segmentation", "48M params", "COCO", ""],["Cascade Mask R-CNN w/ Swin-T Backbone", "object detection + segmentation", "86M params", "COCO", ""]],
#     columns=(['model tested', 'description', 'model size', 'training dataset', 'latency (ms)']))





st.dataframe(df)  # Same as st.write(df)

placeholder = cv2.imread(r"/Users/tushna/Desktop/HAHA.png")

# TODO: Replace this with the original images directory....
temp_images_dir = glob.glob(r"/Users/tushna/Desktop/Classes/Projects/HumanPoseEstimation/lsp_dataset/leeds_sports_trials/*.jpg")

print(len(temp_images_dir))

images_dir = []

for im in temp_images_dir:

    num = im.split("/")[-1]   
    
    if num=="im0114.jpg" or num=="im0048.jpg" or num=="im0101.jpg" or num=="im0111.jpg" or num=="im0041.jpg" or num=="im0167.jpg":
        continue
    if num=="im0065.jpg" or num=="im0070.jpg" or num=="im0015.jpg" or num=="im0032.jpg" or num=="im0097.jpg":
        continue
    if num=="im0019.jpg" or num=="im0080.jpg" or num=="im0109.jpg" or num=="im0055.jpg" or num=="im0087.jpg":
        continue
    if num=="im0078.jpg" or num=="im0050.jpg" or num=="im0092.jpg" or num=="im0161.jpg" or num=="im0163.jpg":
        continue

    images_dir.append(im)

print(len(images_dir))


display_images = []

for num in range(0,10):
    index = random.randint(0, len(images_dir)-1)
    im = images_dir[index]

    split_char = '''/'''
    image_name = im.split(split_char)[-1]
    # num = image_name.split('.')[0]

    display_images.append(image_name)


print(display_images)
# for image in display_images:
#     ## search for this image in all 5 directories,
#     ## and display
#     col1, col2, col3, col4 = st.columns(4)

st.markdown('##')
st.markdown('##')
st.markdown('##')

for i in range(0, len(display_images)):
# for i in range(0, 1):
    # cols = st.columns(4)
    col1, col2, col3 = st.columns(3)
    # col1.write(f'{i}')
    
    num = display_images[i]
    
    with col1:
        placeholder = cv2.imread(r"/Users/tushna/Desktop/Classes/Projects/HumanPoseEstimation/lsp_dataset/leeds_sports_trials/" + num)
        placeholder = cv2.cvtColor(placeholder, cv2.COLOR_RGB2BGR)

        st.image(placeholder, width=200)
        st.subheader("[original]")

        st.markdown('##')
        st.markdown("""---""")
        st.markdown('##')
        st.markdown('##')
        st.markdown('##')
    
    with col2:
        im1_u2 = cv2.imread(r"/Users/tushna/Desktop/Classes/Projects/HumanPoseEstimation/lsp_dataset/leeds_sports_finished/" + num)
        im1_u2 = cv2.cvtColor(im1_u2, cv2.COLOR_RGB2BGR)
        # im1_u2 = im1_u2[:,:,[2,1,0]]

        st.image(im1_u2, width=200)
        st.subheader("[openpose]")
        
        # im1_u2p = cv2.imread(r"C:\Users\t-teduljee\Desktop\done\u2p_masks_applied\\" + num + ".png")
        # im1_u2p = cv2.cvtColor(im1_u2p, cv2.COLOR_BGR2RGB)

        # st.image(placeholder, use_column_width=True)
        # st.subheader("Lightweight U-2 Net")

        st.markdown('##')
        st.markdown("""---""")
        st.markdown('##')
        st.markdown('##')
        st.markdown('##')

    with col3:
        im1_u2p = cv2.imread(r"/Users/tushna/Desktop/Classes/Projects/HumanPoseEstimation/lsp_dataset/alphapose_outputs/" + num)
        im1_u2p = cv2.cvtColor(im1_u2p, cv2.COLOR_RGB2BGR)


        st.image(im1_u2p, width=200)
        st.subheader("[alphapose]")

        # st.image(placeholder, use_column_width=True)
        # st.subheader("Mask R-CCN + Swin-T")
        st.markdown('##')
        st.markdown("""---""")
        st.markdown('##')
        st.markdown('##')
        st.markdown('##')





















































# pic = cv2.imread("/Users/tushna/Desktop/shoe1.png")
# shoe1 = pic
# shoe2 = pic

# images_dir = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

# display_images = []

# for num in range(0,10):
#     index = random.randint(0, len(images_dir)-1)
#     im = images_dir[index]
#     display_images.append(im)

# print(display_images)

# ## display_images is an array of strings, that are image paths
# ## these are the chosen images that we want to now search for
# ## from all models (and take the original too) and display all in one column

# for image in display_images:
#     ## search for this image in all 5 directories,
#     ## and display
#     col1, col2, col3, col4 = st.columns(4)


# # Draw the header and image.
# # st.subheader(header)
# # st.markdown(description)


# c1, c2, c3 = st.columns((1.5,1,1))

# # c4, c5 = st.columns((1,1))

# with c1:
#     st.image(shoe1, use_column_width=True)
#     st.subheader("original")
# with c2:
#     st.image(shoe2, use_column_width=True)
#     st.subheader("u-2 net")
#     st.image(shoe2, use_column_width=True)
#     st.subheader("u-2 net")
# with c3:
#     st.image(shoe1, use_column_width=True)
#     st.subheader("swin T")
#     st.image(shoe1, use_column_width=True)
#     st.subheader("swin T")

    









