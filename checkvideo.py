# from collections import defaultdict
# import cv2
# import numpy as np
# from ultralytics import YOLO
# import time

# # Load the YOLOv8 model
# model = YOLO(r"D:\RaceProject\project\yolov8n.pt")

# # Open the video file
# video_path = r'D:\RaceProject\project\nascars_simple.mp4'
# cap = cv2.VideoCapture(video_path)

# # Store the track history
# track_history = defaultdict(lambda: [])

# # Initialize variables for tracking frame information
# current_frame = 0
# total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# start_time = time.time()

# # Loop through the video frames
# while cap.isOpened():
#     # Read a frame from the video
#     success, frame = cap.read()

#     if success:
#         try:
#             # Increment current frame count
#             current_frame += 1

#             # Run YOLOv8 tracking on the frame, persisting tracks between frames
#             results = model.track(frame, persist=True, classes=2, iou=0.7, conf=0.3, half=True, device='cuda:0',save=True )

#             # Get the boxes and track IDs
#             boxes = results[0].boxes.xywh.cpu()
#             track_ids = results[0].boxes.id.int().cpu().tolist()

#             # Initialize a list to store the bounding box coordinates
#             bounding_boxes = []

#             # Visualize the results on the frame
#             # annotated_frame = results[0].plot()
#             annotated_frame = results[0].plot(labels=False, conf=False)


#             # Plot the tracks and extract bounding box coordinates
#             for box, track_id in zip(boxes, track_ids):
#                 x, y, w, h = box
#                 x1 = int(x - w / 2)
#                 y1 = int(y - h / 2)
#                 x2 = int(x + w / 2)
#                 y2 = int(y + h / 2)
#                 bounding_boxes.append((x1, y1, x2, y2))

#                 track = track_history[track_id]
#                 track.append((float(x), float(y)))  # x, y center point
#                 if len(track) > 30:  # retain 90 tracks for 90 frames
#                     track.pop(0)

#                 # Draw the tracking lines
#                 points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
#                 cv2.polylines(
#                     annotated_frame,
#                     [points],
#                     isClosed=False,
#                     color=(120, 230, 230),
#                     thickness=10,
#                 )

#                 # Calculate elapsed time
#                 elapsed_time = time.time() - start_time

#                 # Print information on the bounding box
#                 info_text = f'{x1},{x2},{y1},{y2},{current_frame}/{total_frames},{track_id},{elapsed_time:.2f}s'
#                 cv2.putText(annotated_frame, info_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

#             # Display the annotated frame
#             cv2.imshow("YOLOv8 Tracking", annotated_frame)

#             # Break the loop if 'q' is pressed
#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 break
#         except:
#             print("Error processing frame")
#             print(boxes)
#             continue
#     else:
#         # Break the loop if the end of the video is reached
#         break

# # Release the video capture object and close the display window
# cap.release()
# cv2.destroyAllWindows()


# from collections import defaultdict
# import cv2
# import numpy as np
# from ultralytics import YOLO
# import time

# # Load the YOLOv8 model
# model = YOLO(r"D:\RaceProject\project\yolov8m.pt")

# # Open the video file
# video_path = r'D:\RaceProject\project\nascars_simple.mp4'
# cap = cv2.VideoCapture(video_path)

# # Store the track history
# track_history = defaultdict(lambda: [])

# # Initialize variables for tracking frame information
# current_frame = 0
# total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# start_time = time.time()

# # Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# output_video_path = 'output_video_2.avi'
# output_video = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# # Loop through the video frames
# while cap.isOpened():
#     # Read a frame from the video
#     success, frame = cap.read()

#     if success:
#         try:
#             # Increment current frame count
#             current_frame += 1

#             # Run YOLOv8 tracking on the frame, persisting tracks between frames
#             results = model.track(frame, persist=True, classes=2, iou=0.7, conf=0.3, half=True, device='cuda:0')

#             # Get the boxes and track IDs
#             boxes = results[0].boxes.xywh.cpu()
#             track_ids = results[0].boxes.id.int().cpu().tolist()

#             # Initialize a list to store the bounding box coordinates
#             bounding_boxes = []

#             # Visualize the results on the frame
#             annotated_frame = results[0].plot(labels=False, conf=False)

#             # Plot the tracks and extract bounding box coordinates
#             for box, track_id in zip(boxes, track_ids):
#                 x, y, w, h = box
#                 x1 = int(x - w / 2)
#                 y1 = int(y - h / 2)
#                 x2 = int(x + w / 2)
#                 y2 = int(y + h / 2)
#                 bounding_boxes.append((x1, y1, x2, y2))

#                 track = track_history[track_id]
#                 track.append((float(x), float(y)))  # x, y center point
#                 if len(track) > 300:  # retain 90 tracks for 90 frames
#                     track.pop(0)

#                 # Draw the tracking lines
#                 points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
#                 cv2.polylines(
#                     annotated_frame,
#                     [points],
#                     isClosed=False,
#                     color=(120, 230, 230),
#                     thickness=10,
#                 )

#                 # Calculate elapsed time
#                 elapsed_time = time.time() - start_time

#                 # Display information on the bounding box
#                 info_text = f'Coordinates: ({x1},{y1}),({x2},{y2})'
#                 cv2.putText(annotated_frame, info_text, (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

#                 info_text = f'Frame: {current_frame}/{total_frames}'
#                 cv2.putText(annotated_frame, info_text, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#                 info_text = f'Car ID: {track_id}'
#                 cv2.putText(annotated_frame, info_text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 192, 203), 2)

#                 info_text = f'Time: {elapsed_time:.2f}s'
#                 cv2.putText(annotated_frame, info_text, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#             # Write the frame with bounding boxes to the output video
#             output_video.write(annotated_frame)

#             # Display the annotated frame
#             cv2.imshow("YOLOv8 Tracking", annotated_frame)

#             # Break the loop if 'q' is pressed
#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 break
#         except Exception as e:
#             print("Error processing frame:", e)
#             print(boxes)
#             continue
#     else:
#         # Break the loop if the end of the video is reached
#         break

# # Release the video capture object and close the display window
# cap.release()
# output_video.release()
# cv2.destroyAllWindows()







from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import time

# Load the YOLOv8 model
model = YOLO(r"D:\RaceProject\project\yolov8x.pt")

# Open the video file
video_path = r'race_real_video.mp4'
# video_path = r'D:\RaceProject\project\nascars_simple.mp4'
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])

# Initialize variables for tracking frame information
current_frame = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
start_time = time.time()

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video_path = 'output_video_3.avi'
output_video = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

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
            # if results[0].boxes.id is not None:
                # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            print(boxes,"boxes")

            try:
                track_ids = results[0].boxes.id.int().cpu().tolist()
                print(track_ids,"track_ids")

                    # Initialize a list to store the bounding box coordinates
                bounding_boxes = []

                    # Visualize the results on the frame
                annotated_frame = results[0].plot(labels=False, conf=False)

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

                        # Draw the tracking lines
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(
                            annotated_frame,
                            [points],
                            isClosed=False,
                            color=(120, 230, 230),
                            thickness=10,
                        )

                        # Calculate elapsed time
                        elapsed_time = time.time() - start_time

                        # Display information on the bounding box
                        info_text = f'Coordinates: ({x1},{y1}),({x2},{y2})'
                        cv2.putText(annotated_frame, info_text, (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                        info_text = f'Frame: {current_frame}/{total_frames}'
                        cv2.putText(annotated_frame, info_text, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                        info_text = f'Car ID: {track_id}'
                        cv2.putText(annotated_frame, info_text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 192, 203), 2)

                        info_text = f'Time: {elapsed_time:.2f}s'
                        cv2.putText(annotated_frame, info_text, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    # Write the frame with bounding boxes to the output video
                output_video.write(annotated_frame)

                    # Display the annotated frame
                cv2.imshow("YOLOv8 Tracking", annotated_frame)

            except:
                 
                 bounding_boxes = []

                # Visualize the results on the frame
                 annotated_frame = results[0].plot(labels=False, conf=False)

                 for box in boxes:
                    x, y, w, h = box
                    x1 = int(x - w / 2)
                    y1 = int(y - h / 2)
                    x2 = int(x + w / 2)
                    y2 = int(y + h / 2)
                    bounding_boxes.append((x1, y1, x2, y2))

                    # Calculate elapsed time
                    elapsed_time = time.time() - start_time

                    # Display information on the bounding box
                    info_text = f'Coordinates: ({x1},{y1}),({x2},{y2})'

                    cv2.putText(annotated_frame, info_text, (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                    info_text = f'Frame: {current_frame}/{total_frames}'

                    cv2.putText(annotated_frame, info_text, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    info_text = f'Time: {elapsed_time:.2f}s'

                    cv2.putText(annotated_frame, info_text, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Write the frame with bounding boxes to the output video

                 output_video.write(annotated_frame)

                # Display the annotated frame
                 cv2.imshow("YOLOv8 Tracking", annotated_frame)
            

                 

            

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        except Exception as e:
            print("Error processing frame:", e)
            continue
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
output_video.release()
cv2.destroyAllWindows()
