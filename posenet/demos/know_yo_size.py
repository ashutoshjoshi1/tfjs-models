import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import json

with open('keypoints.json') as json_file:
    data = json.load(json_file)

pose_df = pd.DataFrame()
for pose in data:
    pose_dict = {}
    for part in pose:
        for cord_name,cord_val in part['position'].items():
            if part['score']>0.1:
                pose_dict[part['part']+'_'+cord_name] = cord_val
            else:
                pose_dict[part['part']+'_'+cord_name] = np.NAN
    pose_df = pose_df.append(pose_dict, ignore_index=True)


Index(['leftAnkle_x', 'leftAnkle_y', 'leftEar_x', 'leftEar_y', 'leftElbow_x',
       'leftElbow_y', 'leftEye_x', 'leftEye_y', 'leftHip_x', 'leftHip_y',
       'leftKnee_x', 'leftKnee_y', 'leftShoulder_x', 'leftShoulder_y',
       'leftWrist_x', 'leftWrist_y', 'nose_x', 'nose_y', 'rightAnkle_x',
       'rightAnkle_y', 'rightEar_x', 'rightEar_y', 'rightElbow_x',
       'rightElbow_y', 'rightEye_x', 'rightEye_y', 'rightHip_x', 'rightHip_y',
       'rightKnee_x', 'rightKnee_y', 'rightShoulder_x', 'rightShoulder_y',
       'rightWrist_x', 'rightWrist_y'],
      dtype='object')


def leng(a, b):
    dist =  math.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2)
    return dist
def get_leng(a,b):
    ab = leng(a,b)
    return ab
def get_dist(pose):
    r_ey = np.array([pose.rightEye_x, pose.rightEye_y])
    r_h = np.array([pose.rightHip_x,pose.rightHip_y])
    r_a = np.array([pose.rightAnkle_x,pose.rightAnkle_y])
    r_s = np.array([pose.rightShoulder_x,pose.rightShoulder_y])
    l_ey = np.array([pose.leftEye_x, pose.leftEye_y])
    l_h = np.array([pose.leftHip_x,pose.leftHip_y])
    l_a = np.array([pose.leftAnkle_x,pose.leftAnkle_y])
    l_s = np.array([pose.leftShoulder_x,pose.leftShoulder_y])
    d_r_ey_a = get_leng(r_ey,r_a)
    R_height.append(d_r_ey_a)
    waist = get_leng(r_h,l_h)
    W_size.append(waist)
    d_l_ey_a = get_leng(l_ey,l_a)
    L_height.append(d_l_ey_a)
    shoulder = get_leng(r_s,l_s)
    S_size.append(shoulder)
    return np.array([d_r_ey_a, waist, d_l_ey_a,shoulder])
R_height , W_size, L_height,S_size = [],[],[],[]
count = 0
eps = []
for _,pose in pose_df.ix[1:].iterrows():
    if _ == 1:
        initial = get_dist(pose)
    else:
        current = get_dist(pose)


RH  = [x for x in R_height if ~np.isnan(x)]
W = [x for x in W_size if ~np.isnan(x)]
LH = [x for x in L_height if ~np.isnan(x)]
S = [x for x in S_size if ~np.isnan(x)]


ARH= sum(RH)/len(RH)
AW = sum(W)/len(W)
ALH = sum(LH)/len(LH)
AH = (ARH + ALH)/2
AS = sum(S)/len(S)


Measure_W = (AW+AH)/AH
Measure_S = (AS+AH)/AH


print("1. Press 1 for Waist Size")
print("2. Press 2 for Shoulder Size")
x = int(input("Enter the option : "))


if x == 1:
    if Measure_W < 1:
        print("Waist Size is 28")
    elif Measure_W >=1 and Measure_W < 1.2:
        print("Waist Size is 30")
    elif Measure_W >=1.2 and Measure_W < 1.4:
        print("Waist size is 32")
    elif Measure_W >=1.4 and Measure_W < 1.6:
        print("Waist size is 34")
    else :
        print("waist size is 36 or plus")

elif x == 2:
    if Measure_S < 1:
        print("shoulder Size is Xtra Small")
    elif Measure_S >=1 and Measure_S < 1.2:
        print("shoulder Size is Small")
    elif Measure_S >=1.2 and Measure_S < 1.4:
        print("shoulder size is Medium")
    elif Measure_S >=1.4 and Measure_S < 1.6:
        print("shoulder size is Large")
    else :
        print("shoulder size is Xtra Large")
