from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import time
import csv
import pika  # Import pika library for RabbitMQ

# Load the YOLOv8 model
model = YOLO(r"D:\RaceProject\project\yolov8m.pt")

# Establish connection with RabbitMQ server
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

# Declare a queue named 'tracking_data'
channel.queue_declare(queue='tracking_data')

# Open the video file
video_path = r'D:\RaceProject\project\nascars_simple.mp4'
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])

# Initialize variables for tracking frame information
current_frame = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
start_time = time.time()

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        try:
            # Increment current frame count
            current_frame += 1

            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True, classes=2, iou=0.7, conf=0.3, half=True, device='cuda:0')

            # Check if detections are present
            if results and results[0].boxes.id is not None:
                # Get the boxes and track IDs
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                # Initialize a list to store the bounding box coordinates
                bounding_boxes = []

                # Plot the tracks and extract bounding box coordinates
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    x1 = int(x - w / 2)
                    y1 = int(y - h / 2)
                    x2 = int(x + w / 2)
                    y2 = int(y + h / 2)
                    bounding_boxes.append((x1, y1, x2, y2))

                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 300:  # retain 90 tracks for 90 frames
                        track.pop(0)

                    # Calculate elapsed time
                    elapsed_time = time.time() - start_time

                    # Prepare the data to be sent to the queue
                    data = {
                        'Frame': current_frame,
                        'X1': x1,
                        'Y1': y1,
                        'X2': x2,
                        'Y2': y2,
                        'Car_ID': track_id,
                        'Time': elapsed_time
                    }

                    # Publish the data to the queue
                    channel.basic_publish(exchange='', routing_key='tracking_data', body=str(data))

        except Exception as e:
            print("Error processing frame:", e)
            continue
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object
cap.release()

# Close the RabbitMQ connection
connection.close()
