import os
import pandas as pd
from sklearn.externals import joblib
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import json

app = Flask(__name__)
cors = CORS(app)

my_json = json.dumps({})

@app.route("/")
def hello():
    return "Hello World!"

@app.route("/secondary2", methods=['POST'])
# def secondary():

# 	# print("API requested")
# 	# keypoint_data = request.get_json()
# 	# print(keypoint_data)

# 	# return my_json
# 	keypoints_json = request.get_json()
# 	print('****************')
# 	print(keypoints_json)
# 	python_dict_to_json_file('../keypoints.json', keypoints_json)

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
def secondary():
    import json
    # Save a python dict object to JSON format file.
    def python_dict_to_json_file(file_path, dict_object):
        try:
            a = []
            if not os.path.isfile(file_path):
                a.append(dict_object)
                with open(file_path, mode='w') as f:
                    f.write(json.dumps(a, indent=2))
            else:
                with open(file_path) as feedsjson:
                    feeds = json.load(feedsjson)

                feeds.append(dict_object)
                with open(file_path, mode='w') as f:
                    f.write(json.dumps(feeds, indent=2))
    
        except FileNotFoundError:
            print(file_path + " not found. ")    

    try:
        keypoints_json = request.get_json()
        print('****************')
        print(keypoints_json)
        python_dict_to_json_file('../keypoints.json', keypoints_json)
        ##############
        #Our code goes here for secondary model




        ###############



    except Exception as e:
        raise e
    responses = jsonify(keypoints=keypoints_json)
    responses.status_code = 200
    return (responses)





if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0', port = 7744)
