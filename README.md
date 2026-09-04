# AI-based-Classroom-Attendance-from-Face-Recognition

## Aim :
To build a system that automatically marks classroom attendance using AI-based face recognition, reducing manual roll call time and improving accuracy.

## System Requirements :
Input: Live camera feed or uploaded classroom photo.

Output: Attendance list with Present/Absent status for each student.

Storage: Database to maintain student profiles and attendance logs.

Interface: Web dashboard for teachers to view, verify, and export attendance reports.

## Features:
Detects faces in classroom images

Matches detected faces with known student images

Automatically marks attendance as Present/Absent

Simple CSV and visual output

## Program :
```
# ============================================================
# FACE RECOGNITION ATTENDANCE SYSTEM
# GOOGLE COLAB
# ============================================================

# NOTE: You will need to install the following libraries first if they are not already present:
!pip install cmake
!pip install dlib
!pip install face_recognition
!pip install opencv-python numpy

import cv2
import face_recognition
import os
import csv
from datetime import datetime
from google.colab.patches import cv2_imshow


# ============================================================
# SETTINGS
# ============================================================

STUDENTS_DIR = "students"
CLASSROOM_IMAGE = "/content/students.jpg" # Ensure this path is correct
ATTENDANCE_FILE = "attendance.csv"

# Lower value = stricter matching
TOLERANCE = 0.50


# ============================================================
# 1. CHECK FILES AND SETUP DIRECTORIES
# ============================================================

print("==========================================")
print("CHECKING FILES AND SETTING UP DIRECTORIES")
print("==========================================")

# Create students directory if it doesn't exist
if not os.path.exists(STUDENTS_DIR):
    os.makedirs(STUDENTS_DIR)
    print(f"[INFO] Created directory: {STUDENTS_DIR}")

# Move student images from content to students directory if they exist
student_images_in_content = [f for f in os.listdir('/content/') if f.lower().endswith(('.jpg', '.jpeg', '.png')) and f in ['charitha.png', 'hasini.png']]
for img_file in student_images_in_content:
    source_path = os.path.join('/content/', img_file)
    destination_path = os.path.join(STUDENTS_DIR, img_file)
    if not os.path.exists(destination_path):
        os.rename(source_path, destination_path)
        print(f"[INFO] Moved {img_file} to {STUDENTS_DIR}/")

print("\nFiles in students folder:")
student_files = os.listdir(STUDENTS_DIR)
if not student_files:
    print("   (No student images found in 'students/' directory. Please upload some!)")
else:
    for file in student_files:
        print(" -", file)


if not os.path.exists(CLASSROOM_IMAGE):
    raise FileNotFoundError(
        f"Classroom image '{CLASSROOM_IMAGE}' not found! Please ensure it's uploaded."
    )

print(f"\n[SUCCESS] Classroom image '{CLASSROOM_IMAGE}' found.")


# ============================================================
# 2. LOAD STUDENT FACES
# ============================================================

print("\n=========================================")
print("LOADING STUDENT FACES")
print("=========================================")

known_faces = []
known_names = []

valid_extensions = (
    ".jpg",
    ".jpeg",
    ".png"
)


for filename in student_files:

    # Ignore non-image files
    if not filename.lower().endswith(
        valid_extensions
    ):
        continue


    path = os.path.join(
        STUDENTS_DIR,
        filename
    )


    print(
        f"\n[INFO] Processing: {filename}"
    )


    try:

        # Load image
        image = face_recognition.load_image_file(
            path
        )


        # Detect face and create encoding
        encodings = face_recognition.face_encodings(
            image
        )


        # Check whether a face was found
        if len(encodings) == 0:

            print(
                f"[WARNING] No face detected in {filename}"
            )

            continue


        # Use first face
        known_faces.append(
            encodings[0]
        )


        # Filename becomes student name
        name = os.path.splitext(
            filename
        )[0]


        known_names.append(name)


        print(
            f"[SUCCESS] Registered: {name}"
        )


    except Exception as e:

        print(
            f"[ERROR] Could not process {filename}"
        )

        print(e)


# ============================================================
# 3. CHECK REGISTERED STUDENTS
# ============================================================

print("\n=========================================")
print("REGISTERED STUDENTS")
print("=========================================")

print(
    f"Total students loaded: {len(known_names)}"
)


for name in known_names:

    print(
        " -",
        name
    )


if len(known_faces) == 0:

    raise ValueError(
        """
No student faces were loaded.

Make sure your student images (e.g., charitha.png, hasini.png) are in the 'students/' directory and contain
clear, visible faces.
"""
    )


# ============================================================
# 4. LOAD CLASSROOM IMAGE
# ============================================================

print("\n=========================================")
print("LOADING CLASSROOM IMAGE")
print("=========================================")

classroom = cv2.imread(
    CLASSROOM_IMAGE
)


if classroom is None:

    raise ValueError(
        f"Unable to read classroom image from '{CLASSROOM_IMAGE}'. Check the path and file integrity."
    )


print(
    "[SUCCESS] Classroom image loaded."
)


# ============================================================
# 5. CONVERT BGR TO RGB
# ============================================================

rgb_classroom = cv2.cvtColor(
    classroom,
    cv2.COLOR_BGR2RGB
)


# ============================================================
# 6. DETECT FACES
# ============================================================

print("\n=========================================")
print("DETECTING FACES")
print("=========================================")

face_locations = face_recognition.face_locations(
    rgb_classroom
)


face_encodings = face_recognition.face_encodings(
    rgb_classroom,
    face_locations
)


print(
    f"[INFO] Detected {len(face_encodings)} face(s)."
)


# ============================================================
# 7. INITIALIZE ATTENDANCE
# ============================================================

present_students = []

absent_students = known_names.copy()


# ============================================================
# 8. RECOGNIZE EACH FACE
# ============================================================

print("\n=========================================")
print("RECOGNIZING STUDENTS")
print("=========================================")


for face_encoding, face_location in zip(
    face_encodings,
    face_locations
):


    # Calculate face distances
    face_distances = face_recognition.face_distance(
        known_faces,
        face_encoding
    )


    # Find closest student
    best_match_index = face_distances.argmin()


    # Default
    name = "Unknown"


    # Check whether closest face is within tolerance
    if face_distances[
        best_match_index
    ] <= TOLERANCE:


        name = known_names[
            best_match_index
        ]


        distance = face_distances[
            best_match_index
        ]


        print(
            f"[MATCH] {name} "
            f"(distance: {distance:.3f})"
        )


        # Mark present
        if name not in present_students:

            present_students.append(
                name
            )


        # Remove from absent
        if name in absent_students:

            absent_students.remove(
                name
            )


    else:

        print(
            "[UNKNOWN] Unknown face detected."
        )


    # ========================================================
    # DRAW RECTANGLE
    # ========================================================

    top, right, bottom, left = face_location


    # Green = recognized
    # Red = unknown

    if name == "Unknown":

        color = (0, 0, 255)

    else:

        color = (0, 255, 0)


    cv2.rectangle(
        classroom,
        (left, top),
        (right, bottom),
        color,
        2
    )


    # ========================================================
    # DISPLAY NAME
    # ========================================================

    cv2.putText(
        classroom,
        name,
        (left, max(top - 10, 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )


# ============================================================
# 9. DISPLAY RESULT
# ============================================================

print("\n=========================================")
print("RECOGNITION RESULT")
print("=========================================")

cv2_imshow(
    classroom
)


# ============================================================
# 10. SAVE ATTENDANCE
# ============================================================

print("\n=========================================")
print("SAVING ATTENDANCE")
print("=========================================")


timestamp = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S"
)


with open(
    ATTENDANCE_FILE,
    "w",
    newline=""
) as file:


    writer = csv.writer(file)


    # CSV header
    writer.writerow(
        [
            "Student Name",
            "Status",
            "Timestamp"
        ]
    )


    # Save each student's status
    for name in known_names:


        if name in present_students:

            status = "Present"

        else:

            status = "Absent"


        writer.writerow(
            [
                name,
                status,
                timestamp
            ]
        )


print(
    f"[SUCCESS] Attendance saved to "
    f"{ATTENDANCE_FILE}"
)


# ============================================================
# 11. ATTENDANCE SUMMARY
# ============================================================

print("\n=========================================")
print("ATTENDANCE SUMMARY")
print("=========================================")


for name in known_names:


    if name in present_students:

        print(
            f"✅ {name}: PRESENT"
        )

    else:

        print(
            f"❌ {name}: ABSENT"
        )


# ============================================================
# 12. STATISTICS
# ============================================================

total_students = len(
    known_names
)

total_present = len(
    present_students
)

total_absent = (
    total_students -
    total_present
)


print("\n=========================================")
print("ATTENDANCE STATISTICS")
print("=========================================")

print(
    "Total Students :",
    total_students
)

print(
    "Present        :",
    total_present
)

print(
    "Absent         :",
    total_absent
)


if total_students > 0:

    percentage = (
        total_present /
        total_students
    ) * 100


    print(
        f"Attendance     : "
        f"{percentage:.2f}%"
    )


print("\n[DONE] Attendance system completed!")
```

## Output:

<img width="965" height="787" alt="image" src="https://github.com/user-attachments/assets/8ca59c95-8eaa-46a1-8c73-4b2b6ed7879c" />

## RESULT:
Thus , to build a system that automatically marks classroom attendance using AI-based face recognition, reducing manual roll call time and improving accuracy has been executed successfully.
