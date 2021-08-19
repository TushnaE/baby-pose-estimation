import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np

from src import model
from src import util
from src.body import Body

body_estimation = Body('model/body_pose_model.pth')


## put a loop here to calculate which frame you're on


ankles_across_frames = []

# for i in range(1,3):
# for i in range(1,61):

    # filename = "images/strokegait_10FPS/frame"+str(i)+".png"

test_image = "images/image.png"
oriImg = cv2.imread(test_image)  # B,G,R order
candidate, subset = body_estimation(oriImg)
canvas = copy.deepcopy(oriImg)
ankles, canvas = util.draw_bodypose(canvas, candidate, subset)

## ankles is a list of tuples --> [(x,y), (x,y)]   where the coordinates are left + right ankles respectively


ankles_across_frames.append(ankles)

# left_ankle = ankles[0]      ## tuple 0
# right_ankle = ankles[1]      ## tuple 1



plt.imshow(canvas[:, :, [2, 1, 0]])
plt.axis('off')
plt.show()


print(ankles_across_frames)

test_image = 'images/strokegait_10FPS/frame20.png'
oriImg = cv2.imread(test_image)  # B,G,R order
candidate, subset = body_estimation(oriImg)
canvas = copy.deepcopy(oriImg)
canvas = util.draw_bodypose(canvas, candidate, subset)

plt.imshow(canvas[:, :, [2, 1, 0]])
plt.axis('off')
plt.show()





## once it works for a thing of 5 images, take each frame away from being 'shown', and do for all images --> put into dictionary and print to console







# import cv2
# import matplotlib.pyplot as plt
# import copy
# import numpy as np

# from src import model
# from src import util
# from src.body import Body

# body_estimation = Body('model/body_pose_model.pth')

# test_image = 'images/distance_estimation/subject_leg.jpg'
# oriImg = cv2.imread(test_image)  # B,G,R order
# candidate, subset = body_estimation(oriImg)
# canvas = copy.deepcopy(oriImg)
# # canvas = util.draw_bodypose(canvas, candidate, subset)

# # plt.imshow(canvas[:, :, [2, 1, 0]])
# plt.axis('off')
# plt.show()
