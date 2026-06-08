import cv2
import numpy as np
import csv
import os

# === Settings ===
video_path = 'VID_0079.mp4'
save_snapshots = False
snapshot_dir = 'frames'
    
csv_path_green  = '20x_test_green.csv'   # full box (with glue)
csv_path_yellow = '20x_test_yellow.csv'  # offset box (without glue)

# Offsets along specimen length (horizontal direction), in pixels
# -> tweak these using the interactive slider script first
LEFT_OFFSET_PX  = 0   # cut away from left
RIGHT_OFFSET_PX = 0   # cut away from right

# === Setup ===
cap = cv2.VideoCapture(video_path)
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
cv2.namedWindow('Mask',  cv2.WINDOW_NORMAL)

if save_snapshots and not os.path.exists(snapshot_dir):
    os.makedirs(snapshot_dir)

csv_file_green  = open(csv_path_green,  mode='w', newline='')
csv_file_yellow = open(csv_path_yellow, mode='w', newline='')

writer_green  = csv.writer(csv_file_green)
writer_yellow = csv.writer(csv_file_yellow)

writer_green.writerow(['Frame', 'Width_green_px', 'Height_green_px'])
writer_yellow.writerow(['Frame', 'Width_yellow_px', 'Height_yellow_px'])

frame_num = 0

# === HSV thresholds ===
lower_green = np.array([0, 158, 62])
upper_green = np.array([179, 255, 255])

# morphology kernel and erode iteration
kernel = np.ones((3, 3), np.uint8)
erode_iter = 3

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (5, 5), 0)

    mask = cv2.inRange(hsv, lower_green, upper_green)

    # morphology
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if erode_iter > 0:
        mask = cv2.erode(mask, kernel, iterations=erode_iter)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    width_green = height_green = 0
    width_yellow = height_yellow = 0

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect).astype(int)

        # --- GREEN BOX (full) ---
        cv2.polylines(frame, [box], isClosed=True, color=(0, 255, 0), thickness=2)

        w, h = rect[1]
        # define width/height as shorter/longer side (for logging)
        if w > h:
            width_green, height_green = int(h), int(w)
        else:
            width_green, height_green = int(w), int(h)

        # --- YELLOW BOX (offset along horizontal / length direction) ---
        pts = box.tolist()
        # sort corners by x to get left/right edges
        pts_sorted_x = sorted(pts, key=lambda p: p[0])
        left_two  = np.array(pts_sorted_x[:2], dtype=float)
        right_two = np.array(pts_sorted_x[2:], dtype=float)

        left_center  = left_two.mean(axis=0)
        right_center = right_two.mean(axis=0)

        vec_lr = right_center - left_center
        len_lr = np.linalg.norm(vec_lr)

        if len_lr > 1e-6:
            # unit vector along specimen length (approximately horizontal)
            u = vec_lr / len_lr
            # perpendicular direction (thickness)
            v = np.array([-u[1], u[0]])

            # project original corners to (s,t) coordinates
            s_vals = []
            t_vals = []
            for p in box:
                p = p.astype(float)
                s_vals.append(np.dot(p, u))
                t_vals.append(np.dot(p, v))

            s_min, s_max = min(s_vals), max(s_vals)
            t_min, t_max = min(t_vals), max(t_vals)

            L0 = s_max - s_min  # original length along u

            left_off  = LEFT_OFFSET_PX
            right_off = RIGHT_OFFSET_PX

            if left_off + right_off < L0 - 1:
                s_min2 = s_min + left_off
                s_max2 = s_max - right_off

                # corrected length
                length_corr = max(int(s_max2 - s_min2), 0)

                # rebuild yellow box corners
                p0 = u * s_min2 + v * t_min
                p1 = u * s_max2 + v * t_min
                p2 = u * s_max2 + v * t_max
                p3 = u * s_min2 + v * t_max

                yellow_box = np.array([p0, p1, p2, p3], dtype=int)
                cv2.polylines(frame, [yellow_box], isClosed=True, color=(0, 255, 255), thickness=2)

                # for logging: width ~ thickness (same as green width), height_yellow = corrected length
                width_yellow  = width_green
                height_yellow = length_corr
            else:
                # offsets too large – nothing left
                width_yellow = 0
                height_yellow = 0

        # --- text overlay (show both) ---
        text_pos = (box[1][0], box[1][1] - 10)
        cv2.putText(frame, f'Green H: {height_green}px',
                    (text_pos[0], text_pos[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f'Yellow H: {height_yellow}px',
                    (text_pos[0], text_pos[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # --- write to CSVs ---
    writer_green.writerow([frame_num, width_green, height_green])
    writer_yellow.writerow([frame_num, width_yellow, height_yellow])

    # --- optional snapshots ---
    if save_snapshots:
        snapshot_path = os.path.join(snapshot_dir, f'frame_{frame_num:04d}.jpg')
        cv2.imwrite(snapshot_path, frame)

    # show windows
    frame_display = cv2.resize(frame, None, fx=0.5, fy=0.5)
    mask_display  = cv2.resize(mask,  None, fx=0.5, fy=0.5)

    cv2.imshow('Frame', frame_display)
    cv2.imshow('Mask',  mask_display)

    if cv2.waitKey(30) & 0xFF == 27:  # ESC
        break

    frame_num += 1

cap.release()
cv2.destroyAllWindows()
csv_file_green.close()
csv_file_yellow.close()

print(f"Green-box measurements saved to {csv_path_green}")
print(f"Yellow-box (offset) measurements saved to {csv_path_yellow}")
if save_snapshots:
    print(f"Frame images saved to {snapshot_dir}/")
