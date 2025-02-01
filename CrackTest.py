import cv2
import numpy as np
import os
from collections import deque
wagon_number = 1
output_folder = "wagon_starting_points"
os.makedirs(output_folder, exist_ok=True)

template_folder = "template"
template_paths = [os.path.join(template_folder, fname) for fname in os.listdir(template_folder) if fname.endswith(".png")]
templates = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in template_paths if cv2.imread(path, cv2.IMREAD_GRAYSCALE) is not None]

if not templates:
    raise ValueError("No valid template images found. Please check the folder.")

video_paths = [
    "Usecase_Dataset/IN/1.mp4", "Usecase_Dataset/IN/1.mp4", "Usecase_Dataset/IN/1.mp4"
]

frame_interval = 25
bottom_crop_ratio = 0.3
min_distance = 150
reset_interval = 200
cooldown_frames = 50

previous_wagons = deque(maxlen=50)

for path in video_paths:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"❌ Error opening video file: {path}")
        continue

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video_path = path.replace(".mp4", "_processed.avi")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    detection_started = False 
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_interval != 0:
            continue

        if frame_count % reset_interval == 0:
            previous_wagons.clear()

        height, width, _ = frame.shape
        cropped_frame = frame[int(height * (1 - bottom_crop_ratio)):, :]
        gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

        all_matches = []
        for template in templates:
            template_height, template_width = template.shape
            result = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.85
            locations = np.where(result >= threshold)

            for pt in zip(*locations[::-1]):
                all_matches.append((pt, result[pt[1], pt[0]]))

        all_matches = sorted(all_matches, key=lambda x: x[1], reverse=True)

        filtered_matches = []
        for match in all_matches:
            pt, confidence = match
            x, y = pt
            overlap = False

            for fm in filtered_matches:
                fx, fy = fm
                if abs(x - fx) < template_width and abs(y - fy) < template_height:
                    overlap = True
                    break

            if not overlap:
                filtered_matches.append((x, y))

        new_wagons = []
        for pt in filtered_matches:
            adjusted_x = pt[0]
            adjusted_y = pt[1] + int(height * (1 - bottom_crop_ratio))
            top_left = (adjusted_x, adjusted_y)
            bottom_right = (top_left[0] + template_width, top_left[1] + template_height)

            is_new = True
            for prev_x, prev_y, prev_frame in previous_wagons:
                if (abs(adjusted_x - prev_x) < min_distance and abs(adjusted_y - prev_y) < min_distance and 
                    frame_count - prev_frame < cooldown_frames):
                    is_new = False
                    break

            if is_new:
                new_wagons.append((adjusted_x, adjusted_y, frame_count))
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

                cropped_wagon = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                output_path = os.path.join(output_folder, f"wagon_{wagon_number}frame{frame_count}.png")

                if cropped_wagon.size > 0:
                    cv2.imwrite(output_path, cropped_wagon)
                    print(f"✅ Wagon {wagon_number} detected in {path} at frame {frame_count}. Saved: {output_path}")

                cv2.putText(frame, f'Wagon {wagon_number}', (adjusted_x, adjusted_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                wagon_number += 1

                detection_started = True

        previous_wagons.extend(new_wagons)

        if detection_started:
            pts = np.array([[int(0.37 * width), 0], [int(0.68 * width), 0],  
                            [int(0.77 * width), height], [int(0.25 * width), height]], np.int32)

            mask = np.zeros_like(frame, dtype=np.uint8)
            cv2.fillPoly(mask, [pts], (255, 255, 255))
            cropped_frame = cv2.bitwise_and(frame, mask)

            gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, threshold1=310, threshold2=350)

            kernel = np.ones((3, 3), np.uint8)
            edges_dilated = cv2.dilate(edges, kernel, iterations=2)
            edges_closed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            damage_mask = np.zeros_like(gray)
            cv2.drawContours(damage_mask, contours, -1, (255), thickness=cv2.FILLED)

            mask_colored = np.zeros_like(frame)
            mask_colored[:, :, 2] = damage_mask 

            overlay = cv2.addWeighted(frame, 1, mask_colored, 0.9, 0)

            out.write(overlay)

            cv2.imshow("Wagon & Damage Detection", overlay)
            if cv2.waitKey(30) & 0xFF == 27:
                break

    cap.release()
    out.release()

cv2.destroyAllWindows()
print("🎯 Processing complete. Videos saved.")