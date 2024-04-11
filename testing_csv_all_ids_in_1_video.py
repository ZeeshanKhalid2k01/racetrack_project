# import cv2
# import pandas as pd
# import numpy as np

# # Load the tracking data from the CSV file
# tracking_data = pd.read_csv('tracking_data.csv')

# # Group the tracking data by Car_ID
# grouped_data = tracking_data.groupby('Car_ID')

# # Generate unique colors for each Car_ID
# color_map = {}
# for idx, (car_id, _) in enumerate(grouped_data):
#     color_map[car_id] = tuple(np.random.randint(0, 255, 3).tolist())

# # Open the video file
# cap = cv2.VideoCapture(r'D:\RaceProject\project\nascars_simple.mp4')

# # Loop through the video frames
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Get the current frame number
#     current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

#     # Draw bounding boxes on the frame
#     for car_id, group in grouped_data:
#         current_frame_data = group[group['Frame'] == current_frame]
#         color = color_map[car_id]
#         for index, row in current_frame_data.iterrows():
#             x1, y1, x2, y2 = int(row['X1']), int(row['Y1']), int(row['X2']), int(row['Y2'])
#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

#     # Display the frame with bounding boxes
#     cv2.imshow('Tracking', frame)

#     # Check for 'q' key press to exit
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         break

# # Release the video capture object
# cap.release()

# # Close all OpenCV windows
# cv2.destroyAllWindows()






import cv2
import pandas as pd
import numpy as np

# Load the tracking data from the CSV file
tracking_data = pd.read_csv('tracking_data.csv')

# Group the tracking data by Car_ID
grouped_data = tracking_data.groupby('Car_ID')

# Generate unique colors for each Car_ID
color_map = {}
for idx, (car_id, _) in enumerate(grouped_data):
    color_map[car_id] = tuple(np.random.randint(0, 255, 3).tolist())

# Open the video file
cap = cv2.VideoCapture(r'D:\RaceProject\project\nascars_simple.mp4')

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video_path = 'output_video_all_ids.avi'
output_video = cv2.VideoWriter(output_video_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

# Loop through the video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get the current frame number
    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    # Draw bounding boxes on the frame
    for car_id, group in grouped_data:
        current_frame_data = group[group['Frame'] == current_frame]
        color = color_map[car_id]
        for index, row in current_frame_data.iterrows():
            x1, y1, x2, y2 = int(row['X1']), int(row['Y1']), int(row['X2']), int(row['Y2'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Write the frame with bounding boxes to the output video
    output_video.write(frame)

# Release the video capture object and close the output video file
cap.release()
output_video.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
