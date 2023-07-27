import cv2,os,face_recognition,time
import numpy as np
import pandas as pd


def detect_faces_in_video(video_path, face_encodings, face_names):
    # Load the pre-trained Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Open the video file
    video = cv2.VideoCapture(0)

    while True:
        # Read a frame from the video
        ret, frame = video.read()
        
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)/4)
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)/4)

        if not ret:
            break

        # Convert the frame to grayscale (face detection requires grayscale images)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        min_size = (frame_width, frame_height)

        print(min_size)
        # Perform face detection on the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.5, minNeighbors=10, minSize=(30, 30))
        # faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=min_size)

        # faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=10, minSize=min_size)

        # Face recognition on detected faces
        for (x, y, w, h) in faces:
            # Crop the detected face region for recognition
            face_image = frame[y:y + h, x:x + w]

            # Convert the cropped face to RGB (face recognition requires RGB images)
            rgb_face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

            # Compute face encodings for the face region
            face_encodings_in_frame = face_recognition.face_encodings(rgb_face_image)

            # Compare face encodings with known face encodings
            recognized_names = []
            for face_encoding in face_encodings_in_frame:
                # Check for a match with known faces
                matches = face_recognition.compare_faces(face_encodings, face_encoding)

                # Find the indexes of recognized faces (if any)
                face_indexes = [i for i, match in enumerate(matches) if match]

                # Get the names of recognized faces
                names = [face_names[i] for i in face_indexes]
                recognized_names.extend(names)

            # Draw rectangles around the detected faces
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Display the list of recognized names
            if recognized_names:
                name_text = ", ".join(recognized_names)
                cv2.putText(frame, name_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            else :
                name_text = "Unknown"
                cv2.putText(frame, name_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Display the output frame
        cv2.imshow('Video Face Detection', frame)

        # Exit the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close the window
    video.release()
    cv2.destroyAllWindows()

def detect_and_recognize_face_in_video(video_input, encoded_faces, student_names, threshold_factor=0.6):
    """
    Recognize faces in a video input and mark attendance for recognized student.

    Args:
        video_input (str): Path to the video input file.
        encoded_faces (list): List of known face encodings.
        student_names (list): List of student names corresponding to the face encodings.
        threshold_factor (float): A factor to adjust the confidence threshold. Default is 0.6.
    """
    capture = cv2.VideoCapture(0)

    while capture.isOpened():
        success, frame = capture.read()
        if not success:
            break

        resized_frame = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
        resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        face_in_resized_frame = face_recognition.face_locations(resized_frame)
        encoded_face_in_resized_frame = face_recognition.face_encodings(resized_frame, face_in_resized_frame)

        # Iterate over detected faces and compare the face encoding with the encodings of known students
        for encoded_face, face_location in zip(encoded_face_in_resized_frame, face_in_resized_frame):
            matches = face_recognition.compare_faces(encoded_faces, encoded_face)
            face_distances = face_recognition.face_distance(encoded_faces, encoded_face)
            match_index = np.argmin(face_distances)

            if matches[match_index] and face_distances[match_index] < threshold_factor:
                name = student_names[match_index].split('.')[0].upper().lower()
                y1, x2, y2, x1 = face_location
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

                # Draw bounding box and write the name
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (x1 + 6, y2 - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow('Video', frame)
                cv2.waitKey(2000)
            else:
                y1, x2, y2, x1 = face_location
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, "Unknown", (x1 + 6, y2 - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow('Video', frame)
                cv2.waitKey(2000)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    known_images_dir = "../ImagesVideos/known_faces"
    image_files = os.listdir(known_images_dir)

    face_encodings = []
    face_names = []

    for image_file in image_files:
        # Construct the absolute file path for the current image
        image_path = os.path.join(known_images_dir, image_file)

        # Load the image file
        image = face_recognition.load_image_file(image_path)

        # Compute face encodings for the image
        encodings = face_recognition.face_encodings(image)

        # If there are no faces found in the image, skip it
        if len(encodings) == 0:
            continue

        # Assuming there's only one face per image, we'll store the first encoding
        face_encodings.append(encodings[0])

        # Store the corresponding face name (you can extract it from the image_file if needed)
        face_names.append(image_file)

    video_path = '../ImagesVideos/video/spoof10.mp4'
    time1 = int(time.time())
    # detect_faces_in_video(video_path,face_encodings,face_names)
    detect_and_recognize_face_in_video(video_path,face_encodings,face_names)
    time2 = int(time.time())
    print(time2-time1)
    print("Finished")
