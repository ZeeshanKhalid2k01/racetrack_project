import csv
import os
import pika

# Callback function to handle incoming messages
def callback(ch, method, properties, body):
    # Convert the message body from string to dictionary
    data = eval(body.decode('utf-8'))
    
    # CSV file path
    csv_file_path = 'tracking_data.csv'

    # Check if CSV file exists
    file_exists = os.path.exists(csv_file_path)

    # Open the CSV file in append mode and create a CSV writer object
    with open(csv_file_path, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        # If the file doesn't exist, write the headers
        if not file_exists:
            csv_writer.writerow(['Frame', 'X1', 'Y1', 'X2', 'Y2', 'Car_ID', 'Time'])

        # Write the data to the CSV file
        csv_writer.writerow([data['Frame'], data['X1'], data['Y1'], data['X2'], data['Y2'], data['Car_ID'], data['Time']])

    print("Received and saved message:", data)

# Establish connection with RabbitMQ server
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

# Declare a queue named 'tracking_data'
channel.queue_declare(queue='tracking_data')

# Set up a consumer and start consuming messages
channel.basic_consume(queue='tracking_data', on_message_callback=callback, auto_ack=True)

print('Waiting for messages. To exit press CTRL+C')

# Start consuming messages
channel.start_consuming()
