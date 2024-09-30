import cv2
from ultralytics import YOLO


plate_model = YOLO('fine_tuned_model.pt')


def image_cut(img, x, y, w, h):
    return img[y:h, x:w]


def process_video_file(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    while cap.isOpened():
        success, frame = cap.read()

        if success:

            results = plate_model(frame)
            annotated_frame = results[0].plot()

            cv2.imshow("YOLOv8 Inference", annotated_frame)

            if results[0].boxes.conf.nelement() > 0 and results[0].boxes.conf[0].item() > 0.90:
                x, y, w, h = map(int, results[0].boxes.data[0][:4])
                img = image_cut(frame, x, y, w, h)


            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            print("End of video or error reading frame.")
            break

    cap.release()
    cv2.destroyAllWindows()



process_video_file('WhatsApp Video 2024-09-29 at 20.50.00_315a350f.mp4',)
