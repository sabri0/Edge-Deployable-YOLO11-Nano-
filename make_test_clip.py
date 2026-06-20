"""Generate a tiny synthetic SLR clip so the pipeline can be exercised end-to-end
without the (access-restricted) UCO dataset. Renders a simple stick figure lying
supine and lifting the right leg, so YOLO11-pose has a human-like shape to detect.
This is a smoke-test input only, not clinical data."""
import cv2
import numpy as np

W, H, N = 640, 480, 90
out = cv2.VideoWriter("test_slr.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))

for i in range(N):
    frame = np.full((H, W, 3), 30, np.uint8)
    phase = np.sin(np.pi * i / N)            # 0 -> 1 -> 0 leg lift
    # supine person, head left, feet right
    shoulder = (180, 240)
    hip = (340, 240)
    knee = (430 - int(40 * phase), 240 - int(90 * phase))
    ankle = (520 - int(80 * phase), 240 - int(180 * phase))
    head = (150, 240)
    cv2.circle(frame, head, 22, (200, 200, 200), -1)
    for a, b in [(head, shoulder), (shoulder, hip), (hip, knee), (knee, ankle)]:
        cv2.line(frame, a, b, (200, 200, 200), 14)
    for p in [shoulder, hip, knee, ankle]:
        cv2.circle(frame, p, 8, (180, 180, 180), -1)
    out.write(frame)

out.release()
print("wrote test_slr.mp4", N, "frames")
