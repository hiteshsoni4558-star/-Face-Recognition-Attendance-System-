import cv2
import numpy as np
import face_recognition
import os
import csv
from datetime import datetime

class FaceAttendanceSystem:
    def __init__(self):
        # Initialize paths
        self.images_path = "images"
        self.attendance_file = "attendance.csv"
        
        # Create images directory if it doesn't exist
        if not os.path.exists(self.images_path):
            os.makedirs(self.images_path)
            
        # Create attendance file with header if it doesn't exist
        if not os.path.exists(self.attendance_file):
            with open(self.attendance_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Name", "Date", "Time"])
    
    def capture_image(self, name):
        """Capture and save user's face image"""
        print(f"Capturing image for {name}...")
        print("Press SPACE to capture or ESC to cancel")
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        while True:
            success, img = cap.read()
            if not success:
                print("Failed to access webcam.")
                break
                
            # Display the image
            cv2.imshow("Webcam", img)
            
            # Wait for key press
            key = cv2.waitKey(1)
            
            # Space key for capture
            if key == 32:  # SPACE key
                user_dir = os.path.join(self.images_path, name)
                if not os.path.exists(user_dir):
                    os.makedirs(user_dir)
                
                image_path = os.path.join(user_dir, f"/images/image.jpg")
                cv2.imwrite(image_path, img)
                print(f"Image saved to {image_path}")
                break
            
            # ESC key to cancel
            elif key == 27:  # ESC key
                print("Capture canceled")
                break
        
        # Release webcam and close windows
        cap.release()
        cv2.destroyAllWindows()
        
    def load_images(self):
        """Load all images and create face encodings"""
        print("Loading registered faces...")
        
        # Lists to hold face data
        known_face_encodings = []
        known_face_names = []
        
        # Scan through the images directory
        for person_name in os.listdir(self.images_path):
            person_dir = os.path.join(self.images_path, person_name)
            if os.path.isdir(person_dir):
                for image_file in os.listdir(person_dir):
                    image_path = os.path.join(person_dir, image_file)
                    if os.path.isfile(image_path):
                        # Load the image
                        face_image = face_recognition.load_image_file(image_path)
                        
                        # Try to find face encodings
                        face_encodings = face_recognition.face_encodings(face_image)
                        
                        if face_encodings:
                            # Use the first face found in the image
                            face_encoding = face_encodings[0]
                            known_face_encodings.append(face_encoding)
                            known_face_names.append(person_name)
                            print(f"Loaded face for: {person_name}")
                        else:
                            print(f"No face found in image: {image_path}")
        
        return known_face_encodings, known_face_names
        
    def mark_attendance(self, name):
        """Mark attendance in the CSV file"""
        now = datetime.now()
        date_string = now.strftime("%Y-%m-%d")
        time_string = now.strftime("%H:%M:%S")
        
        # Check if person already marked attendance today
        already_marked = False
        if os.path.exists(self.attendance_file):
            with open(self.attendance_file, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    if len(row) >= 2 and row[0] == name and row[1] == date_string:
                        already_marked = True
                        break
        
        # If not marked today, add entry
        if not already_marked:
            with open(self.attendance_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([name, date_string, time_string])
                print(f"Attendance marked for {name}")
        else:
            print(f"{name} already marked attendance today")
            
    def recognize_faces(self):
        """Start face recognition process using webcam"""
        print("Starting face recognition...")
        print("Press ESC to exit")
        
        # Load known faces
        known_face_encodings, known_face_names = self.load_images()
        
        if not known_face_encodings:
            print("No registered faces found. Please register a face first.")
            return
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        # Set for tracking recognized faces to avoid multiple recognitions
        recognized_faces = set()
        
        while True:
            success, img = cap.read()
            if not success:
                print("Failed to access webcam.")
                break
                
            # Resize frame for faster processing
            small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
            
            # Convert from BGR to RGB (face_recognition uses RGB)
            rgb_small_frame = small_frame[:, :, ::-1]
            
            # Find faces in current frame
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            # Process each face found
            for face_encoding, face_location in zip(face_encodings, face_locations):
                # Compare with known faces
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                
                # Use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    
                    # Mark attendance if face recognized and not already marked
                    if name not in recognized_faces:
                        self.mark_attendance(name)
                        recognized_faces.add(name)
                
                # Draw rectangle and name around face
                top, right, bottom, left = face_location
                # Scale back up face locations
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                # Draw rectangle
                cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Draw name
                cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Display the resulting image
            cv2.imshow("Face Recognition", img)
            
            # Exit on ESC key
            if cv2.waitKey(1) == 27:  # ESC key
                break
        
        # Release webcam and close windows
        cap.release()
        cv2.destroyAllWindows()
    
    def delete_user(self, name):
        """Delete a registered user"""
        user_dir = os.path.join(self.images_path, name)
        
        if os.path.exists(user_dir):
            # Remove user directory and all its contents
            for file in os.listdir(user_dir):
                os.remove(os.path.join(user_dir, file))
            os.rmdir(user_dir)
            print(f"User {name} deleted successfully.")
        else:
            print(f"User {name} not found.")
    
    def list_users(self):
        """List all registered users"""
        users = []
        
        if os.path.exists(self.images_path):
            for item in os.listdir(self.images_path):
                user_dir = os.path.join(self.images_path, item)
                if os.path.isdir(user_dir):
                    users.append(item)
        
        return users

def main():
    system = FaceAttendanceSystem()
    
    while True:
        print("\n===== Face Recognition Attendance System =====")
        print("1. Register a new face")
        print("2. Start face recognition & mark attendance")
        print("3. Delete a registered user")
        print("4. List registered users")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == '1':
            name = input("Enter name of the person: ")
            system.capture_image(name)
        
        elif choice == '2':
            system.recognize_faces()
        
        elif choice == '3':
            users = system.list_users()
            
            if not users:
                print("No registered users found.")
                continue
                
            print("\nRegistered users:")
            for i, user in enumerate(users, 1):
                print(f"{i}. {user}")
                
            user_choice = input("\nEnter the number of the user to delete (or 0 to cancel): ")
            
            try:
                user_idx = int(user_choice) - 1
                if user_idx == -1:  # User entered 0
                    continue
                
                if 0 <= user_idx < len(users):
                    system.delete_user(users[user_idx])
                else:
                    print("Invalid selection.")
            except ValueError:
                print("Please enter a valid number.")
        
        elif choice == '4':
            users = system.list_users()
            
            if users:
                print("\nRegistered users:")
                for i, user in enumerate(users, 1):
                    print(f"{i}. {user}")
            else:
                print("No registered users found.")
        
        elif choice == '5':
            print("Exiting the program...")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()# -Face-Recognition-Attendance-System-
