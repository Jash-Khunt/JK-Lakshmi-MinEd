import cv2
import torch
import numpy as np

midas = torch.hub.load('intel-isl/MiDaS', 'DPT_Large')
midas.to('cuda')
midas.eval()

transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform

input_video_path = r"D:\\Pawan\\Jupyter Notebook\\mined\\Usecase_Dataset\\Out\\2.mp4"

cap = cv2.VideoCapture(input_video_path)

wagon_length_real = 12.7
wagon_width_real = 2.8

estimated_volume = 0
frame_counter = 0
wagon_counter = 0
frames_since_last_wagon = 0
wagon_volumes = []
wagon_in_frame = False
wagon_threshold = 30

def is_wagon_filled(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = frame.shape[0] * frame.shape[1]
    filled_area = sum(cv2.contourArea(cnt) for cnt in contours)
    fill_ratio = filled_area / total_area
    return fill_ratio > 0.06

def get_depth_map(frame):
    imgbatch = transform(frame).to('cuda')
    with torch.no_grad():
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode='bicubic',
            align_corners=True
        ).squeeze()
    return prediction.cpu().numpy()

prev_frame = None
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    frames_since_last_wagon += 1

    if is_wagon_filled(frame):
        depth_map = get_depth_map(frame)
        depth_map = np.nan_to_num(depth_map, nan=0.0)
        depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_map = np.uint8(depth_map)

        height, width = depth_map.shape
        pixel_scale_x = wagon_length_real / width
        pixel_scale_y = wagon_width_real / height
        pixel_area = pixel_scale_x * pixel_scale_y

        depth_values = depth_map.astype(np.float32) * pixel_area
        frame_volume = np.sum(depth_values)

        if prev_frame is not None:
            movement_score = np.mean(cv2.absdiff(prev_frame, depth_map))
            if movement_score > 2:
                if not wagon_in_frame:
                    if wagon_counter > 0:
                        avg_volume = estimated_volume / frames_since_last_wagon
                        print(f"Wagon {wagon_counter} processed. Avg Volume: {avg_volume:.2f} m³")
                        wagon_volumes.append(avg_volume)
                    
                    print("New Wagon Detected!")
                    wagon_counter += 1
                    estimated_volume = 0
                    frames_since_last_wagon = 0
                wagon_in_frame = True
                estimated_volume += frame_volume
        prev_frame = depth_map.copy()
    else:
        wagon_in_frame = False

cap.release()
cv2.destroyAllWindows()

if wagon_volumes:
    print(f"Total Wagons Detected: {wagon_counter}")
    print(f"Average Volume Per Wagon: {np.mean(wagon_volumes):.2f} m³")
else:
    print("No wagons detected.")