import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin


app = Flask(__name__)
cors = CORS(app)

# pose_df = pd.DataFrame()


#=====[ Returns true if the specified index of y_coords is a minimum value ]=====
def is_min(y_coords, height, gradient, index, epsilon, beta):
    if np.abs(y_coords[index] - height)/height < epsilon:
        for i in range(1,beta):
            if gradient[index - i] > 0 or gradient[index + i - 1] < 0:
                return False
        return True
        
#=====[ Returns true if we suspect that we are in a new repetition ]=====
def in_new_squat(y_coords, height, index, delta):
    return abs((y_coords[index] - height)/height) > delta

#=====[ Gets local maxes within accepted epsilon of global max and with max len(y_coors)/gamma maxes ]=====
#=====[ For squats: usual epsilon ~ 0.2, gamma ~ 20, delta ~ 0.5, beta ~ 1 ]=====
#=====[ For pushups: usual epsilon ~ 0.2, gamma ~ 20, delta ~ 0.2, beta ~ 1 ]=====

def get_local_mins(y_coords, epsilon=0.2, gamma=50, delta=0.2, beta=2):
    
    local_mins = []
    height = np.min(y_coords[int(len(y_coords)/3):int(len(y_coords)*2/3)])
    gradient = np.gradient(y_coords)
    
    #=====[ Checks gradients to make sure we are looking at a local min ]=====
    min_located = False
    for index, dy in enumerate(gradient[2:]):
        if(min_located):
            if in_new_squat(y_coords, height, index, delta):
                min_located = False       
            else:
                continue
                
        if  is_min(y_coords, height, gradient, index, epsilon, beta + 1):
            local_mins.append(index)
            min_located = True
        
    return sorted(local_mins)


@app.route("/")
def hello():
    return "Hello World!"


@app.route("/secondary", methods=['POST'])
def secondary():
    try:
        keypoints_json = request.get_json()
        print('****************')
        # print(keypoints_json)
        # python_dict_to_json_file('../keypoints.json', keypoints_json)
        try:
            pose_df = pd.read_csv('../pose_df.csv')
        except:
            pose_df = pd.DataFrame()
        pose_dict = keypoints_json
        print(pose_dict)
        pose_df = pose_df.append(pose_dict, ignore_index=True)
        # print(pose_df)
        pose_df.to_csv('../pose_df.csv', index=False)  


    except Exception as e:
        raise e
    resp_dic={'count':1}
    resp = jsonify(resp_dic)
    resp.status_code = 200
    return (resp)

if __name__ == '__main__':
    df = pd.DataFrame()
    df.to_csv('../pose_df.csv', index=False)
    app.run(debug=True, processes=8)

# @app.route("/secondary", methods=['POST'])
# def secondary():
#     try:
#         keypoints_json = request.get_json()
#         print('****************')
#         # print(keypoints_json)
#         # python_dict_to_json_file('../keypoints.json', keypoints_json)
#         pose = keypoints_json
#         try:
#             pose_df = pd.read_csv('../pose_df.csv')
#         except:
#             pose_df = pd.DataFrame()
#         pose_dict = {}
#         for part in pose:
#             for cord_name,cord_val in part['position'].items():
#                 if part['score']>0.1:
#                     pose_dict[part['part']+'_'+cord_name] = cord_val
#                 else:
#                     pose_dict[part['part']+'_'+cord_name] = np.NAN
        
#         y_coords = np.array(pose_df['leftEar_y'])
#         if len(y_coords)%50 ==0:
#             pose_dict['count'] = len(get_local_mins(y_coords))
#         else:
#             pose_dict['count'] = max(pose_df['leftEar_y'])
        
#         pose_df = pose_df.append(pose_dict, ignore_index=True)
        
#         # print(pose_df)
#         pose_df.to_csv('../pose_df.csv', index=False)  


#     except Exception as e:
#         raise e
#     # responses = jsonify(keypoints=keypoints_json)
#     resp_dic={'count':pose_dict['count']}
#     resp = jsonify(resp_dic)
#     resp.status_code = 200
#     return (resp)

# if __name__ == '__main__':
#     cols = {'leftAnkle_x':None, 'leftAnkle_y':None, 'leftEar_x':None, 'leftEar_y':None, 'leftElbow_x':None,
#        'leftElbow_y':None, 'leftEye_x':None, 'leftEye_y':None, 'leftHip_x':None, 'leftHip_y':None,
#        'leftKnee_x':None, 'leftKnee_y':None, 'leftShoulder_x':None, 'leftShoulder_y':None,
#        'leftWrist_x':None, 'leftWrist_y':None, 'nose_x':None, 'nose_y':None, 'rightAnkle_x':None,
#        'rightAnkle_y':None, 'rightEar_x':None, 'rightEar_y':None, 'rightElbow_x':None,
#        'rightElbow_y':None, 'rightEye_x':None, 'rightEye_y':None, 'rightHip_x':None, 'rightHip_y':None,
#        'rightKnee_x':None, 'rightKnee_y':None, 'rightShoulder_x':None, 'rightShoulder_y':None,
#        'rightWrist_x':None, 'rightWrist_y':None, 'count':0}
#     df = pd.DataFrame()
#     df.to_csv('../pose_df.csv', index=False)
#     app.run(debug=True, processes=8)


    # import json
    # # Save a python dict object to JSON format file.

    # def plot_graph(pose_df):
    #     plt.figure(figsize=(10,8))
    #     plt.plot(pose_df.index, pose_df.leftEye_y, label='left eye')
    #     plt.plot(pose_df.index, pose_df.rightEye_y, label='right eye')
    #     plt.xlabel('time')
    #     plt.ylabel('eye movement')
    #     plt.legend()
    #     plt.savefig('../graph.png')
    #     plt.clf()

    # def python_dict_to_json_file(file_path, dict_object):
    #     try:
    #         a = []
    #         if not os.path.isfile(file_path):
    #             a.append(dict_object)
    #             with open(file_path, mode='w') as f:
    #                 f.write(json.dumps(a, indent=2))
    #         else:
    #             with open(file_path) as feedsjson:
    #                 feeds = json.load(feedsjson)

    #             feeds.append(dict_object)
    #             with open(file_path, mode='w') as f:
    #                 f.write(json.dumps(feeds, indent=2))
    
    #     except FileNotFoundError:
    #         print(file_path + " not found. ")    
