import os
import face_recognition
import cv2

# Path to the folder containing images of people
abd_folder = r"abd"  # Replace with the actual path to the folder
aiman_folder =  r"aiman"
anas_folder = r"anas"
arshi_folder = r"arshi"
bhaskar_folder = r"bhaskar"
coo_folder =  r"coo"
ehraz_folder = r"ehraz"
fateh_folder = r"fateh"
himani_folder = r"himani"
raghav_folder = r"raghav"
samridhi_folder = r"samridhi"
soni_folder = r"soni"
yash_folder = r"yash"
# Load known faces and their names
known_faces = {
    "Yash": [os.path.join(yash_folder, file_name) for file_name in os.listdir(yash_folder)],
     "Soni": [os.path.join(soni_folder, file_name) for file_name in os.listdir(soni_folder)],
     "Samridhi": [os.path.join(samridhi_folder, file_name) for file_name in os.listdir(samridhi_folder)],
    "Fateh": [os.path.join(fateh_folder, file_name) for file_name in os.listdir(fateh_folder)],
    "Ehraz": [os.path.join(ehraz_folder, file_name) for file_name in os.listdir(ehraz_folder)],
     "Coo": [os.path.join(coo_folder, file_name) for file_name in os.listdir(coo_folder)],
     "Himani": [os.path.join(himani_folder, file_name) for file_name in os.listdir(himani_folder)],
     "Bhaskar": [os.path.join(bhaskar_folder, file_name) for file_name in os.listdir(bhaskar_folder)],
 "Arshi": [os.path.join(arshi_folder, file_name) for file_name in os.listdir(arshi_folder)],
     "Anas": [os.path.join(anas_folder, file_name) for file_name in os.listdir(anas_folder)],
     "Aiman": [os.path.join(aiman_folder, file_name) for file_name in os.listdir(aiman_folder)],
     "Abdullah": [os.path.join(abd_folder, file_name) for file_name in os.listdir(abd_folder)],
     "Raghav": [os.path.join(raghav_folder, file_name) for file_name in os.listdir(raghav_folder)],

    
    }

# Initialize face encodings and names
known_face_encodings = []
known_face_names = []

for name, image_paths in known_faces.items():
    person_encodings = []
    for image_path in image_paths:
        image = face_recognition.load_image_file(image_path)
        
        if len(face_recognition.face_encodings(image))>0:
            encoding = face_recognition.face_encodings(image)[0]  # Assuming one face per image
            person_encodings.append(encoding)
    known_face_encodings.append(person_encodings)
    known_face_names.append(name)

# Initialize video capture
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    # Find faces in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations,model='cnn')

    # Loop through each detected face
    for face_encoding in face_encodings:
        # Compare face encoding with known faces
        matches = [face_recognition.compare_faces(person_encodings, face_encoding,tolerance=0.5) for person_encodings in known_face_encodings]
        name = "Unknown"

        # Check if there's a matchqq
        for idx, person_matches in enumerate(matches):
            if True in person_matches:
                name = known_face_names[idx]
                break

        # Draw rectangle around the face and label it
        top, right, bottom, left = face_recognition.face_locations(frame)[0]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Exit loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
video_capture.release()
cv2.destroyAllWindows()
