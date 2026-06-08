import cv2
import numpy as np

video_path = 'VID_0079.mp4'

cap = cv2.VideoCapture(video_path)

cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)

def nothing(x):
    pass

# --- HSV sliders ---
cv2.createTrackbar('H_min', 'Controls', 40, 179, nothing)
cv2.createTrackbar('H_max', 'Controls', 80, 179, nothing)
cv2.createTrackbar('S_min', 'Controls', 80, 255, nothing)
cv2.createTrackbar('S_max', 'Controls', 255, 255, nothing)
cv2.createTrackbar('V_min', 'Controls', 80, 255, nothing)
cv2.createTrackbar('V_max', 'Controls', 255, 255, nothing)

# morphology
cv2.createTrackbar('Erode', 'Controls', 0, 3, nothing)

# offsets along *horizontal/length* direction (pixels)
cv2.createTrackbar('Left_off',  'Controls', 0, 300, nothing)
cv2.createTrackbar('Right_off', 'Controls', 0, 300, nothing)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (5, 5), 0)

    # read sliders
    h_min = cv2.getTrackbarPos('H_min', 'Controls')
    h_max = cv2.getTrackbarPos('H_max', 'Controls')
    s_min = cv2.getTrackbarPos('S_min', 'Controls')
    s_max = cv2.getTrackbarPos('S_max', 'Controls')
    v_min = cv2.getTrackbarPos('V_min', 'Controls')
    v_max = cv2.getTrackbarPos('V_max', 'Controls')
    erode_iter = cv2.getTrackbarPos('Erode', 'Controls')

    left_off  = cv2.getTrackbarPos('Left_off',  'Controls')
    right_off = cv2.getTrackbarPos('Right_off', 'Controls')

    lower_green = np.array([h_min, s_min, v_min])
    upper_green = np.array([h_max, s_max, v_max])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if erode_iter > 0:
        mask = cv2.erode(mask, kernel, iterations=erode_iter)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect).astype(int)

        # draw original min-area box (green)
        cv2.polylines(frame, [box], isClosed=True, color=(0, 255, 0), thickness=2)

        # width/height = shorter/longer side of rect (just for info)
        w, h = rect[1]
        if w > h:
            width, height = int(h), int(w)
        else:
            width, height = int(w), int(h)

        # ---------- Build yellow box trimmed LEFT/RIGHT (horizontal) ----------
        # 1) Get left/right edge midpoints to define horizontal axis
        pts = box.tolist()
        pts_sorted_x = sorted(pts, key=lambda p: p[0])  # sort by x
        left_two  = np.array(pts_sorted_x[:2], dtype=float)
        right_two = np.array(pts_sorted_x[2:], dtype=float)

        left_center = left_two.mean(axis=0)
        right_center = right_two.mean(axis=0)

        vec_lr = right_center - left_center
        len_lr = np.linalg.norm(vec_lr)

        if len_lr > 1e-6:
            # unit vector along length (horizontal-ish) and its perpendicular
            u = vec_lr / len_lr
            v = np.array([-u[1], u[0]])  # perpendicular

            # 2) Project original box corners to (s,t) coordinates
            s_vals = []
            t_vals = []
            for p in box:
                p = p.astype(float)
                s_vals.append(np.dot(p, u))
                t_vals.append(np.dot(p, v))

            s_min, s_max = min(s_vals), max(s_vals)
            t_min, t_max = min(t_vals), max(t_vals)

            L0 = s_max - s_min  # full length along u

            # 3) Apply offsets along length
            if left_off + right_off < L0 - 1:  # ensure something remains
                s_min2 = s_min + left_off
                s_max2 = s_max - right_off

                # corrected effective length
                length_corr = int(s_max2 - s_min2)
                if length_corr < 0:
                    length_corr = 0

                # 4) Rebuild yellow box corners from (s,t)
                p0 = u * s_min2 + v * t_min
                p1 = u * s_max2 + v * t_min
                p2 = u * s_max2 + v * t_max
                p3 = u * s_min2 + v * t_max

                yellow_box = np.array([p0, p1, p2, p3], dtype=int)
                cv2.polylines(frame, [yellow_box], isClosed=True, color=(0, 255, 255), thickness=2)

                # Label (using corrected length as "Height corr" since that's your specimen length)
                text_pos = (box[1][0], box[1][1] - 10)
                cv2.putText(frame,
                            f'Width: {width}px',
                            (text_pos[0], text_pos[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)
                cv2.putText(frame,
                            f'Height corr: {length_corr}px',
                            (text_pos[0], text_pos[1] + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 255), 2)

    # show smaller windows
    frame_display = cv2.resize(frame, None, fx=0.5, fy=0.5)
    mask_display = cv2.resize(mask, None, fx=0.5, fy=0.5)

    cv2.imshow('Frame', frame_display)
    cv2.imshow('Mask', mask_display)

    if cv2.waitKey(30) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
