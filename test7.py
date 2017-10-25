import face_recognition
import cv2
import numpy
import os
import time

video_capture = cv2.VideoCapture(0)

files = os.listdir("known_people")
del files[0]
print(files)

training_face_encoding=[]

for file in files:
    training_image = face_recognition.load_image_file("known_people/"+file)
    training_face_encoding.append(face_recognition.face_encodings(training_image)[0])

def train_known_people():
    print("train_known_people")
    files = os.listdir("known_people")
    del files[0]
    print(files)

    training_face_encoding=[]
    for file in files:
        print(file)
        training_image = face_recognition.load_image_file("known_people/"+file)
        training_face_encoding.append(face_recognition.face_encodings(training_image)[0])
    recalculate = False
    return

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

#recalculate variable used when new people is recognize
recalculate = False

i = 0
while True:
    if recalculate:
        print("Recalculate is needed")
        files = os.listdir("known_people")
        del files[0]
        training_face_encoding=[]
        for file in files:
            print("\t\t\t\t"+file)
            training_image = face_recognition.load_image_file("known_people/"+file)
            training_face_encoding.append(face_recognition.face_encodings(training_image)[0])
        recalculate = False
        process_this_frame = True
    else:
        ret, frame = video_capture.read()
        frame=cv2.flip(frame, 1)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        if i < 4:
            process_this_frame=False
            i += 1
        else:
            i = 0

        if process_this_frame:
            face_locations = face_recognition.face_locations(small_frame)
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)
            
            face_names = []
            face_encode = 0
            for face_encoding in face_encodings:
                face_recognition.compare_faces(training_face_encoding, face_encoding)
                distance = face_recognition.face_distance(training_face_encoding, face_encoding)
                name = "Unknown"

                for distance_files in zip(distance, files):
                    print("distance, people:\t '{0}'".format(distance_files))
                
                distance_value = numpy.amin(distance)
                distance_index = numpy.argmin(distance)
                
                print("distance value:\t\t %s" % (distance_value))
                print("distance_index:\t\t %s" % (distance_index))
                
                if distance_value < 0.6:
                    name = files[distance_index]
                else:
                    #need to get this crop of image of unknown person
                    #print("NEED TO GET IMAGE OF UNKNOWN PERSON")
                    print("crop")
                    name = "unknown_"+str(time.time())
                    ttop = (face_locations[face_encode][0]*4)-15
                    tright = (face_locations[face_encode][1]*4)+15
                    tleft = (face_locations[face_encode][3]*4)-15
                    tbottom = (face_locations[face_encode][2]*4)+15
                    crop_img = frame[(ttop+2):(tbottom-35), (tleft+2):(tright-2)]
                    cv2.imwrite("known_people/"+name+".jpg", crop_img)

                    go_ahed = False
                    while not go_ahed:
                        try:
                            print("tring")
                            cv2.imread("known_people/"+name+".jpg", 0)
                            go_ahed = True
                        except OSError:
                            print("saving")
                    recalculate = True
                
                print("name_matched:\t\t %s \n" % (name))
                face_names.append(name)
                face_encode += 1

        process_this_frame = not process_this_frame

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            top -= 15
            left -= 15
            right += 15
            bottom +=15
            
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video_capture.release()
cv2.destroyAllWindows()
