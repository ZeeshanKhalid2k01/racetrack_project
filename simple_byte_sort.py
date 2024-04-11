from ultralytics import YOLO

model =YOLO(r"D:\RaceProject\project\yolov8x.pt")

results=model.track(source=r'race_real_video.mp4', show=True, tracker='bytetrack.yaml', save=True, classes=2, iou=0.7, conf=0.3, half=True, device='cuda:0')
# results=model.track(source=r'D:\RaceProject\project\nascars_simple.mp4', show=True, tracker='botsort.yaml', save=True, classes=2)