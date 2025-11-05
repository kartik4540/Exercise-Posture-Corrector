import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import datetime
import gradio as gr
import sklearn
import pickle
import warnings
import math

warnings.filterwarnings('ignore')

# Drawing helpers
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# ============================================
# BICEP CURL - Helper Functions and Classes
# ============================================

IMPORTANT_LMS1 = [
    "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"
]

headersbc = ["label"]
for lm in IMPORTANT_LMS1:
    headersbc += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]


def extract_important_keypoints(results) -> list:
    landmarks = results.pose_landmarks.landmark
    data = []
    for lm in IMPORTANT_LMS1:
        keypoint = landmarks[mp_pose.PoseLandmark[lm].value]
        data.append([keypoint.x, keypoint.y, keypoint.z, keypoint.visibility])
    return np.array(data).flatten().tolist()


def calculate_angle(point1: list, point2: list, point3: list) -> float:
    point1 = np.array(point1)
    point2 = np.array(point2)
    point3 = np.array(point3)
    angleInRad = np.arctan2(point3[1] - point2[1], point3[0] - point2[0]) - np.arctan2(point1[1] - point2[1], point1[0] - point2[0])
    angleInDeg = np.abs(angleInRad * 180.0 / np.pi)
    angleInDeg = angleInDeg if angleInDeg <= 180 else 360 - angleInDeg
    return angleInDeg


class BicepPoseAnalysis:
    def __init__(self, side: str, stage_down_threshold: float, stage_up_threshold: float, 
                 peak_contraction_threshold: float, loose_upper_arm_angle_threshold: float, visibility_threshold: float):
        self.stage_down_threshold = stage_down_threshold
        self.stage_up_threshold = stage_up_threshold
        self.peak_contraction_threshold = peak_contraction_threshold
        self.loose_upper_arm_angle_threshold = loose_upper_arm_angle_threshold
        self.visibility_threshold = visibility_threshold
        self.side = side
        self.counter = 0
        self.stage = "down"
        self.is_visible = True
        self.detected_errors = {"LOOSE_UPPER_ARM": 0, "PEAK_CONTRACTION": 0}
        self.loose_upper_arm = False
        self.peak_contraction_angle = 1000
        self.peak_contraction_frame = None
    
    def get_joints(self, landmarks) -> bool:
        side = self.side.upper()
        joints_visibility = [
            landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].visibility,
            landmarks[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].visibility,
            landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].visibility
        ]
        is_visible = all([vis > self.visibility_threshold for vis in joints_visibility])
        self.is_visible = is_visible
        if not is_visible:
            return self.is_visible
        
        self.shoulder = [landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].x, 
                        landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].y]
        self.elbow = [landmarks[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].x, 
                     landmarks[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].y]
        self.wrist = [landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].x, 
                     landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].y]
        return self.is_visible
    
    def analyze_pose(self, landmarks, frame):
        self.get_joints(landmarks)
        if not self.is_visible:
            return (None, None)
        
        bicep_curl_angle = int(calculate_angle(self.shoulder, self.elbow, self.wrist))
        if bicep_curl_angle > self.stage_down_threshold:
            self.stage = "down"
        elif bicep_curl_angle < self.stage_up_threshold and self.stage == "down":
            self.stage = "up"
            self.counter += 1
        
        shoulder_projection = [self.shoulder[0], 1]
        ground_upper_arm_angle = int(calculate_angle(self.elbow, self.shoulder, shoulder_projection))
        
        if ground_upper_arm_angle > self.loose_upper_arm_angle_threshold:
            if not self.loose_upper_arm:
                self.loose_upper_arm = True
                self.detected_errors["LOOSE_UPPER_ARM"] += 1
        else:
            self.loose_upper_arm = False
        
        if self.stage == "up" and bicep_curl_angle < self.peak_contraction_angle:
            self.peak_contraction_angle = bicep_curl_angle
            self.peak_contraction_frame = frame
        elif self.stage == "down":
            if self.peak_contraction_angle != 1000 and self.peak_contraction_angle >= self.peak_contraction_threshold:
                self.detected_errors["PEAK_CONTRACTION"] += 1
            self.peak_contraction_angle = 1000
            self.peak_contraction_frame = None
        
        return (bicep_curl_angle, ground_upper_arm_angle)


# ============================================
# SQUAT - Helper Functions
# ============================================

imp_lnd = ["NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP",
           "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"]

headersquat = ["label"]
for lm in imp_lnd:
    headersquat += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]


def extract_important_keypoints_sq(results) -> list:
    landmark = results.pose_landmarks.landmark
    data = []
    for lm in imp_lnd:
        keypoint = landmark[mp_pose.PoseLandmark[lm].value]
        data.append([keypoint.x, keypoint.y, keypoint.z, keypoint.visibility])
    return np.array(data).flatten().tolist()


def calculate_distance(pointX, pointY) -> float:
    x1, y1 = pointX
    x2, y2 = pointY
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def analyze_foot_knee_placement(results, stage: str, foot_shoulder_ratio_thresholds: list, 
                                knee_foot_ratio_thresholds: dict, visibility_threshold: int) -> dict:
    analyzed_results = {"foot_placement": -1, "knee_placement": -1}
    landmark = results.pose_landmarks.landmark
    
    left_foot_index_vis = landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].visibility
    right_foot_index_vis = landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].visibility
    left_knee_vis = landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility
    right_knee_vis = landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility
    
    if (left_foot_index_vis < visibility_threshold or right_foot_index_vis < visibility_threshold or 
        left_knee_vis < visibility_threshold or right_knee_vis < visibility_threshold):
        return analyzed_results
    
    left_shoulder = [landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                    landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    right_shoulder = [landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
                     landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    shoulder_width = calculate_distance(left_shoulder, right_shoulder)
    
    left_foot_index = [landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x, 
                      landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
    right_foot_index = [landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x, 
                       landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
    foot_width = calculate_distance(left_foot_index, right_foot_index)
    foot_shoulder_ratio = round(foot_width / shoulder_width, 1)
    
    min_ratio_foot_shoulder, max_ratio_foot_shoulder = foot_shoulder_ratio_thresholds
    if min_ratio_foot_shoulder <= foot_shoulder_ratio <= max_ratio_foot_shoulder:
        analyzed_results["foot_placement"] = 0
    elif foot_shoulder_ratio < min_ratio_foot_shoulder:
        analyzed_results["foot_placement"] = 1
    elif foot_shoulder_ratio > max_ratio_foot_shoulder:
        analyzed_results["foot_placement"] = 2
    
    left_knee = [landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].x, 
                landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    right_knee = [landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, 
                 landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    knee_width = calculate_distance(left_knee, right_knee)
    knee_foot_ratio = round(knee_width / foot_width, 1)
    
    if stage == "up":
        up_min, up_max = knee_foot_ratio_thresholds.get("up")
        if up_min <= knee_foot_ratio <= up_max:
            analyzed_results["knee_placement"] = 0
        elif knee_foot_ratio < up_min:
            analyzed_results["knee_placement"] = 1
        elif knee_foot_ratio > up_max:
            analyzed_results["knee_placement"] = 2
    elif stage == "middle":
        mid_min, mid_max = knee_foot_ratio_thresholds.get("middle")
        if mid_min <= knee_foot_ratio <= mid_max:
            analyzed_results["knee_placement"] = 0
        elif knee_foot_ratio < mid_min:
            analyzed_results["knee_placement"] = 1
        elif knee_foot_ratio > mid_max:
            analyzed_results["knee_placement"] = 2
    elif stage == "down":
        down_min, down_max = knee_foot_ratio_thresholds.get("down")
        if down_min <= knee_foot_ratio <= down_max:
            analyzed_results["knee_placement"] = 0
        elif knee_foot_ratio < down_min:
            analyzed_results["knee_placement"] = 1
        elif knee_foot_ratio > down_max:
            analyzed_results["knee_placement"] = 2
    
    return analyzed_results


# Load Squat Model
with open("squat_model/model/LR_model.pkl", "rb") as f:
    count_model = pickle.load(f)


# ============================================
# PUSHUP - Helper Functions
# ============================================

imp_pu = [
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER", "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
    "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT", "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY",
    "LEFT_INDEX", "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB", "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
]

headerspu = ["label"]
for lm in imp_pu:
    headerspu += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]


def extract_important_keypoints_pu(results) -> list:
    landmarks = results.pose_landmarks.landmark
    data = []
    for lm in imp_pu:
        keypoint = landmarks[mp_pose.PoseLandmark[lm].value]
        data.append([keypoint.x, keypoint.y, keypoint.z, keypoint.visibility])
    return np.array(data).flatten().tolist()


# Load Pushup Models
with open("pushup_model/pickles/RF_model.pkl", "rb") as f:
    sklearn_model_pu = pickle.load(f)

with open("pushup_model/pickles/input_scaler.pkl", "rb") as f2:
    input_scaler_pu = pickle.load(f2)


def get_class(prediction: float) -> str:
    return {0: "C", 1: "H", 2: "L"}.get(prediction)


# Note: Deep learning model skipped due to Keras version compatibility
# with open("pushup_model/pickles/pushup_dp.pkl", "rb") as f:
#     DL_model_pu = pickle.load(f)


# ============================================
# PLANK - Helper Functions
# ============================================

IMP_LMS = [
    "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
]

headersplank = ["label"]
for lm in IMP_LMS:
    headersplank += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]


def extract_important_keypoints_pl(results) -> list:
    landmarks = results.pose_landmarks.landmark
    data = []
    for lm in IMP_LMS:
        keypoint = landmarks[mp_pose.PoseLandmark[lm].value]
        data.append([keypoint.x, keypoint.y, keypoint.z, keypoint.visibility])
    return np.array(data).flatten().tolist()


# Load Plank Models
with open("plank_model/model/LR_model.pkl", "rb") as f:
    sklearn_model_pl = pickle.load(f)

with open("plank_model/model/input_scaler.pkl", "rb") as f2:
    input_scaler_pl = pickle.load(f2)


def get_class_plank(prediction: float) -> str:
    return {0: "C", 1: "H", 2: "L"}.get(prediction)


# ============================================
# ANALYSIS FUNCTIONS
# ============================================

# Bicep Analysis Constants
VISIBILITY_THRESHOLD_BICEP = 0.65
STAGE_UP_THRESHOLD_BICEP = 90
STAGE_DOWN_THRESHOLD_BICEP = 120
PEAK_CONTRACTION_THRESHOLD_BICEP = 60
LOOSE_UPPER_ARM_ANGLE_THRESHOLD_BICEP = 40

left_arm_analysis = BicepPoseAnalysis(
    side="left", stage_down_threshold=STAGE_DOWN_THRESHOLD_BICEP,
    stage_up_threshold=STAGE_UP_THRESHOLD_BICEP,
    peak_contraction_threshold=PEAK_CONTRACTION_THRESHOLD_BICEP,
    loose_upper_arm_angle_threshold=LOOSE_UPPER_ARM_ANGLE_THRESHOLD_BICEP,
    visibility_threshold=VISIBILITY_THRESHOLD_BICEP
)

right_arm_analysis = BicepPoseAnalysis(
    side="right", stage_down_threshold=STAGE_DOWN_THRESHOLD_BICEP,
    stage_up_threshold=STAGE_UP_THRESHOLD_BICEP,
    peak_contraction_threshold=PEAK_CONTRACTION_THRESHOLD_BICEP,
    loose_upper_arm_angle_threshold=LOOSE_UPPER_ARM_ANGLE_THRESHOLD_BICEP,
    visibility_threshold=VISIBILITY_THRESHOLD_BICEP
)


def analyze_bicep_pose(video):
    cap = cv2.VideoCapture(video)
    results_summary = []

    with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose:
        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                (left_bicep_curl_angle, _) = left_arm_analysis.analyze_pose(landmarks=landmarks, frame=image)
                (right_bicep_curl_angle, _) = right_arm_analysis.analyze_pose(landmarks=landmarks, frame=image)

                left_bicep_curl_angle = left_bicep_curl_angle if left_bicep_curl_angle is not None else 0.0
                right_bicep_curl_angle = right_bicep_curl_angle if right_bicep_curl_angle is not None else 0.0

                result_summary = (
                    f"Left Angle: {left_bicep_curl_angle:.2f}, Right Angle: {right_bicep_curl_angle:.2f}, "
                    f"Left Counter: {left_arm_analysis.counter}, Right Counter: {right_arm_analysis.counter}"
                )
                results_summary.append(result_summary)

    cap.release()
    summary_text = "\n".join(results_summary[-10:]) if results_summary else "No data"
    return summary_text


# Squat Analysis Constants
PREDICTION_PROB_THRESHOLD_SQUAT = 0.7
VISIBILITY_THRESHOLD_SQUAT = 0.6
FOOT_SHOULDER_RATIO_THRESHOLDS = [1.2, 2.8]
KNEE_FOOT_RATIO_THRESHOLDS = {
    "up": [0.5, 1.0],
    "middle": [0.7, 1.0],
    "down": [0.7, 1.1],
}


def analyze_squat_pose(video):
    cap = cv2.VideoCapture(video)
    counter = 0
    current_stage = ""
    results_summary = []

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if not results.pose_landmarks:
                continue

            try:
                row = extract_important_keypoints_sq(results)
                X = pd.DataFrame([row], columns=headersquat[1:])

                predicted_class = count_model.predict(X)[0]
                predicted_class = "down" if predicted_class == 0 else "up"
                prediction_probabilities = count_model.predict_proba(X)[0]
                prediction_probability = round(prediction_probabilities[prediction_probabilities.argmax()], 2)

                if predicted_class == "down" and prediction_probability >= PREDICTION_PROB_THRESHOLD_SQUAT:
                    current_stage = "down"
                elif current_stage == "down" and predicted_class == "up" and prediction_probability >= PREDICTION_PROB_THRESHOLD_SQUAT: 
                    current_stage = "up"
                    counter += 1

                analyzed_results = analyze_foot_knee_placement(
                    results=results, stage=current_stage, 
                    foot_shoulder_ratio_thresholds=FOOT_SHOULDER_RATIO_THRESHOLDS, 
                    knee_foot_ratio_thresholds=KNEE_FOOT_RATIO_THRESHOLDS, 
                    visibility_threshold=VISIBILITY_THRESHOLD_SQUAT
                )

                foot_placement_evaluation = analyzed_results["foot_placement"]
                knee_placement_evaluation = analyzed_results["knee_placement"]
                
                foot_placement = ["Correct", "Too tight", "Too wide", "UNK"][foot_placement_evaluation] if foot_placement_evaluation != -1 else "UNK"
                knee_placement = ["Correct", "Too tight", "Too wide", "UNK"][knee_placement_evaluation] if knee_placement_evaluation != -1 else "UNK"

                results_summary.append({
                    "counter": counter,
                    "stage": current_stage,
                    "prediction_probability": prediction_probability,
                    "foot_placement": foot_placement,
                    "knee_placement": knee_placement
                })

            except Exception as e:
                print(f"Error: {e}")

    cap.release()
    
    if results_summary:
        summary_text = "\n".join([
            f"Count: {result['counter']}, Probability: {result['prediction_probability']}, "
            f"Foot: {result['foot_placement']}, Knee: {result['knee_placement']}" 
            for result in results_summary[-10:]
        ])
    else:
        summary_text = "No data"
    
    return summary_text


# Pushup Analysis
PREDICTION_PROB_THRESHOLD_PUSHUP = 0.6


def analyze_pushup_pose(video):
    cap = cv2.VideoCapture(video)  
    results_summary = []

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if not results.pose_landmarks:
                results_summary.append("No human found")
                continue

            try:
                row = extract_important_keypoints_pu(results)
                X = pd.DataFrame([row], columns=headerspu[1:])
                X = pd.DataFrame(input_scaler_pu.transform(X))

                # Using sklearn model
                predicted_class_pu = sklearn_model_pu.predict(X)[0]
                prediction_probability_pu = sklearn_model_pu.predict_proba(X)[0].max()

                if predicted_class_pu == 0 and prediction_probability_pu >= PREDICTION_PROB_THRESHOLD_PUSHUP:
                    current_stage_pu = "Correct"
                elif predicted_class_pu == 2 and prediction_probability_pu >= PREDICTION_PROB_THRESHOLD_PUSHUP:
                    current_stage_pu = "Low back"
                elif predicted_class_pu == 1 and prediction_probability_pu >= PREDICTION_PROB_THRESHOLD_PUSHUP:
                    current_stage_pu = "High back"
                else:
                    current_stage_pu = "Unknown"

                results_summary.append(f"Stage: {current_stage_pu}, Probability: {prediction_probability_pu:.2f}")

            except Exception as e:
                results_summary.append(f"Error: {e}")

    cap.release()
    return "\n".join(results_summary[-10:]) if results_summary else "No data"


# Plank Analysis
PREDICTION_PROB_THRESHOLD_PLANK = 0.6


def analyze_plank_pose(video):
    cap = cv2.VideoCapture(video)
    output_text = []

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if not results.pose_landmarks:
                output_text.append("No human found")
                continue

            try:
                row = extract_important_keypoints_pl(results)
                X = pd.DataFrame([row], columns=headersplank[1:])
                X = pd.DataFrame(input_scaler_pl.transform(X))

                predicted_class = sklearn_model_pl.predict(X)[0]
                predicted_class = get_class_plank(predicted_class)
                prediction_probability = sklearn_model_pl.predict_proba(X)[0]
                prob_value = round(prediction_probability[np.argmax(prediction_probability)], 2)

                if predicted_class == "C" and prediction_probability[prediction_probability.argmax()] >= PREDICTION_PROB_THRESHOLD_PLANK:
                    current_stage = "Correct"
                elif predicted_class == "L" and prediction_probability[prediction_probability.argmax()] >= PREDICTION_PROB_THRESHOLD_PLANK: 
                    current_stage = "Low back"
                elif predicted_class == "H" and prediction_probability[prediction_probability.argmax()] >= PREDICTION_PROB_THRESHOLD_PLANK: 
                    current_stage = "High back"
                else:
                    current_stage = "Unknown"

                output_text.append(f"Stage: {current_stage}, Probability: {prob_value}")

            except Exception as e:
                output_text.append(f"Error: {e}")
        
    cap.release()
    return "\n".join(output_text[-10:]) if output_text else "No data"


# ============================================
# GRADIO INTERFACE
# ============================================

with gr.Blocks() as demo:
    gr.Markdown("# Exercise Posture Corrector")
    gr.Markdown("Upload videos to analyze your exercise form for Bicep Curls, Squats, Pushups, and Planks")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Bicep Curl Analysis")
            bicep_input = gr.Video(label="Upload Video")
            bicep_output = gr.Textbox(label="Results", interactive=False, lines=10)
            bicep_button = gr.Button("Analyze Bicep Curl")
            bicep_button.click(analyze_bicep_pose, inputs=bicep_input, outputs=bicep_output)

        with gr.Column():
            gr.Markdown("## Squat Analysis")
            squat_input = gr.Video(label="Upload Video")
            squat_output = gr.Textbox(label="Results", interactive=False, lines=10)
            squat_button = gr.Button("Analyze Squat")
            squat_button.click(analyze_squat_pose, inputs=squat_input, outputs=squat_output)

    with gr.Row():
        with gr.Column():
            gr.Markdown("## Pushup Analysis")
            pushup_input = gr.Video(label="Upload Video")
            pushup_output = gr.Textbox(label="Results", interactive=False, lines=10)
            pushup_button = gr.Button("Analyze Pushup")
            pushup_button.click(analyze_pushup_pose, inputs=pushup_input, outputs=pushup_output)

        with gr.Column():
            gr.Markdown("## Plank Analysis")
            plank_input = gr.Video(label="Upload Video")
            plank_output = gr.Textbox(label="Results", interactive=False, lines=10)
            plank_button = gr.Button("Analyze Plank")
            plank_button.click(analyze_plank_pose, inputs=plank_input, outputs=plank_output)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
