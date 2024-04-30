from flask import Flask, request, jsonify, Response
import cv2 as cv
import numpy as np
import opensim as osim
import os
import tempfile
import math


app = Flask(__name__)

# Load the OpenPose BODY_25 model files
protoFile = "body_25/pose_deploy.prototxt"
weightsFile = "body_25/pose_iter_584000.caffemodel"
net = cv.dnn.readNetFromCaffe(protoFile, weightsFile)

# Define body parts as per the BODY_25 model
BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "MidHip": 8, "RHip": 9,
    "RKnee": 10, "RAnkle": 11, "LHip": 12, "LKnee": 13, "LAnkle": 14,
    "REye": 15, "LEye": 16, "REar": 17, "LEar": 18, "LBigToe": 19,
    "LSmallToe": 20, "LHeel": 21, "RBigToe": 22, "RSmallToe": 23, "RHeel": 24,
    "Background": 25
}

# Calculate angle function
def calculate_angle(p1, p2, p3):
    if p1 is None or p2 is None or p3 is None:
        return None  # Return None if any point is missing
    a = (p1[0] - p2[0], p1[1] - p2[1])
    b = (p3[0] - p2[0], p3[1] - p2[1])
    dot_product = a[0] * b[0] + a[1] * b[1]
    magnitude_1 = math.sqrt(a[0]**2 + a[1]**2)
    magnitude_2 = math.sqrt(b[0]**2 + b[1]**2)
    if magnitude_1 * magnitude_2 == 0:
        return None  # Avoid division by zero
    angle = math.acos(dot_product / (magnitude_1 * magnitude_2))
    return math.degrees(angle)

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read the image from file storage
    filestr = file.read()
    npimg = np.frombuffer(filestr, np.uint8)
    frame = cv.imdecode(npimg, cv.IMREAD_COLOR)
    return jsonify(process_frame(frame))

def process_frame(frame):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    inpBlob = cv.dnn.blobFromImage(frame, 1.0 / 255, (368, 368), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    out = net.forward()

    points = {}
    for i, part in enumerate(BODY_PARTS.keys()):
        # Slice heatmap of corresponding body's part.
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = int((frameWidth * point[0]) / out.shape[3])
        y = int((frameHeight * point[1]) / out.shape[2])
        
        if conf > 0.2:
            points[part] = (x, y)
        else:
            points[part] = None

    # Calculate angles of interest
    angles = {}
    angle_points = {
        'RElbow': ('RShoulder', 'RElbow', 'RWrist'),
        'LElbow': ('LShoulder', 'LElbow', 'LWrist'),
        'RKnee': ('RHip', 'RKnee', 'RAnkle'),
        'LKnee': ('LHip', 'LKnee', 'LAnkle'),
        'RHip': ('RShoulder', 'RHip', 'RKnee'),
        'LHip': ('LShoulder', 'LHip', 'LKnee'),
        'RAnkleFlexion': ('RKnee', 'RAnkle', 'RHeel'),
        'LAnkleFlexion': ('LKnee', 'LAnkle', 'LHeel'),
        'RToeFlexion': ('RAnkle', 'RBigToe', 'RSmallToe'),
        'LToeFlexion': ('LAnkle', 'LBigToe', 'LSmallToe'),
        'RFootArch': ('RHeel', 'RBigToe', 'RSmallToe'),
        'LFootArch': ('LHeel', 'LBigToe', 'LSmallToe')
    }
    for key, (p1, p2, p3) in angle_points.items():
        angles[key] = calculate_angle(points.get(p1), points.get(p2), points.get(p3))

    return {'points': points, 'angles': angles}


@app.route('/process_opensim', methods=['POST'])
def process_opensim():
    if 'model_file' not in request.files or 'trc_file' not in request.files:
        return jsonify({'error': 'Missing files'}), 400

    model_file = request.files['model_file']
    trc_file = request.files['trc_file']

    # Use temporary directories to ensure proper cleanup
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'model_file.osim')
            trc_path = os.path.join(temp_dir, 'trc_file.trc')

            model_file.save(model_path)
            trc_file.save(trc_path)

            angles = calculate_joint_angles(model_path, trc_path)
            return jsonify(angles)
    except Exception as e:
        app.logger.error(f"Error processing files: {str(e)}")
        return jsonify({'error': str(e)}), 500

def calculate_joint_angles(model_path, trc_path):
    try:
        model = osim.Model(model_path)
        model.initSystem()

        ik_tool = osim.InverseKinematicsTool()
        ik_tool.setModel(model)
        ik_tool.setMarkerDataFileName(trc_path)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mot') as tmp_output:
            output_motion_file = tmp_output.name
            ik_tool.setOutputMotionFileName(output_motion_file)

        marker_set = model.getMarkerSet()
        for i in range(marker_set.getSize()):
            marker = marker_set.get(i)
            ik_task = osim.IKMarkerTask()
            ik_task.setName(marker.getName())
            ik_task.setWeight(1.0)
            ik_tool.getIKTaskSet().adoptAndAppend(ik_task)

        ik_tool.run()

        storage = osim.Storage(output_motion_file)
        angles = process_motion_storage(storage, model)

        os.remove(output_motion_file)

        return angles
    except Exception as e:
        raise Exception(f"Failed to calculate joint angles: {str(e)}")

def process_motion_storage(storage, model):
    try:
        model.initSystem()
        joint_angles_with_time = {}
        
        # Retrieve the time array from storage
        time_array = osim.ArrayDouble()
        storage.getTimeColumn(time_array)
        times = [time_array.get(j) for j in range(time_array.getSize())]
        
        for i in range(model.getCoordinateSet().getSize()):
            coord = model.getCoordinateSet().get(i)
            joint_name = coord.getName()
            angle_series = osim.ArrayDouble()
            storage.getDataColumn(joint_name, angle_series)
            angles = [angle_series.get(j) for j in range(angle_series.getSize())]

            # Pair each time with its corresponding angle
            joint_angles_with_time[joint_name] = list(zip(times, angles))
            
        return joint_angles_with_time
    except Exception as e:
        raise Exception(f"Failed to process motion storage: {str(e)}")



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
