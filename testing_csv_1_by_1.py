import cv2
import pandas as pd

# Load the tracking data from the CSV file
tracking_data = pd.read_csv('tracking_data.csv')

# Group the tracking data by Car_ID
grouped_data = tracking_data.groupby('Car_ID')

# Loop through each Car_ID
for car_id, group in grouped_data:
    # Open the video file
    cap = cv2.VideoCapture(r'D:\RaceProject\project\nascars_simple.mp4')

    # Loop through the video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get the current frame number
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # Get the tracking data for the current Car_ID and frame
        current_frame_data = group[group['Frame'] == current_frame]

        # Draw bounding boxes on the frame
        # Draw bounding boxes on the frame
        for index, row in current_frame_data.iterrows():
            x1, y1, x2, y2 = int(row['X1']), int(row['Y1']), int(row['X2']), int(row['Y2'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


        # Display the frame with bounding boxes
        cv2.imshow('Tracking', frame)

        # Check for 'q' key press to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Release the video capture object
    cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()


















# import cv2
# import pandas as pd
# import os

# # Load the tracking data from the CSV file
# tracking_data = pd.read_csv('tracking_data.csv')

# # Group the tracking data by Car_ID
# grouped_data = tracking_data.groupby('Car_ID')

# # Create a folder to store individual car recordings
# os.makedirs('car_recording', exist_ok=True)

# # Loop through each Car_ID
# for car_id, group in grouped_data:
#     # Open the video file
#     cap = cv2.VideoCapture(r'D:\RaceProject\project\nascars_simple.mp4')

#     # Define the codec and create VideoWriter object for MP4
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(f'car_recording/{car_id}_recording.mp4', fourcc, 30, (int(cap.get(3)), int(cap.get(4))))

#     # Loop through the video frames
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Get the current frame number
#         current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

#         # Get the tracking data for the current Car_ID and frame
#         current_frame_data = group[group['Frame'] == current_frame]

#         # Draw bounding boxes on the frame
#         for index, row in current_frame_data.iterrows():
#             x1, y1, x2, y2 = int(row['X1']), int(row['Y1']), int(row['X2']), int(row['Y2'])
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#         # Write frame to VideoWriter
#         out.write(frame)

#     # Release the video capture object and VideoWriter
#     cap.release()
#     out.release()

# # Close all OpenCV windows
# cv2.destroyAllWindows()
