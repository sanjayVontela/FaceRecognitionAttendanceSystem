# FaceRecognitionAttendanceSystem
FACE-RECOGNITION-ATTENDANCE-SYSTEM

A Python script that recognizes faces and marks attendance in the database. You need to install the following libraries on your system.

1.	Tensorflow/Keras
2.	Opencv
3.	Psycopg2
4.	Pillow

How to use:
1.	Save images of people with their names in their respective folders.
2.	The name of the folder should be their name.
3.	Run datapreprocessing.py to weight faces and save them locally.
4.	Run facerecognition.py file to recognize faces. It uses files stored by datapreprocessing.py 
5.	Attendance is recorded for recognized faces and stored in the database.
6.	2 database files one is for sqlite3 and the other is for psycopg2 depending upon your interest.

