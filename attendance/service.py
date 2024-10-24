from attendance import constants
from PIL import Image

from attendance.models import AttendanceModel, StudentModel
import numpy as np
import cv2
import os
import time

from datetime import datetime

def capture_img(userid):

    try:
        cam = cv2.VideoCapture(0)
        detector = cv2.CascadeClassifier(constants.path + 'haarcascade_frontalface_default.xml')
        sampleNum = 0

        while (True):

            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # incrementing sample number
                sampleNum = sampleNum + 1

                imgloc=constants.dataset_path+"TrainingImages/"+userid + '.' + str(sampleNum) + ".jpg"

                 # saving the captured face in the dataset folder

                cv2.imwrite(imgloc,gray[y:y + h, x:x + w])
                cv2.imshow('Frame', img)

            # wait for 100 miliseconds
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 100
            elif sampleNum > 70:
                break

        cam.release()
        cv2.destroyAllWindows()

    except Exception as e:
            print(e)
    return

def trainimg():

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    global detector
    detector = cv2.CascadeClassifier(constants.path +"haarcascade_frontalface_default.xml")
    try:
        global faces,Id
        faces, Id = getImagesAndLabels(constants.dataset_path+"TrainingImages")
        print("IDS List",Id)
        recognizer.train(faces, np.array(Id))
    except Exception as e:
        print(e)

    try:
        recognizer.save(constants.dataset_path+"TrainingImageLabel\\Trainner.yml")
    except Exception as e:
        print(e)

    return

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # create empth face list
    faceSamples = []
    # create empty ID list
    Ids = []
    # now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        # loading the image and converting it to gray scale
        pilImage = Image.open(imagePath).convert('L')
        # Now we are converting the PIL image into numpy array
        imageNp = np.array(pilImage, 'uint8')
        # getting the Id from the image

        Id = int(os.path.split(imagePath)[-1].split(".")[0])
        # extract the face from the training image sample
        faces = detector.detectMultiScale(imageNp)
        # If a face is there then append that in the list as well as Id of it
        for (x, y, w, h) in faces:
            faceSamples.append(imageNp[y:y + h, x:x + w])
            Ids.append(Id)
            print("Label:",str(Ids))
    return faceSamples, Ids

def attendanceIn():

    now = time.time()
    future = now + 20

    if time.time() < future:
        recognizer = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer()
        try:
            recognizer.read(constants.dataset_path+"TrainingImageLabel\\Trainner.yml")
        except Exception as e:
            print(e)

        harcascadePath = constants.path+"haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(harcascadePath)
        cam = cv2.VideoCapture(0)
        time.sleep(2.0)
        font = cv2.FONT_HERSHEY_SIMPLEX

        while True:
            ret, im = cam.read()
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.2, 5)
            for (x, y, w, h) in faces:
                Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
                if (conf < 70):
                    print(conf)
                    cv2.rectangle(im, (x, y), (x + w, y + h), (0, 260, 0), 7)

                    td = datetime.now()
                    cd=str(td.day)+"-"+str(td.month)+"-"+str(td.year)
                    cr=td.hour

                    student=StudentModel.objects.filter(rno=Id).first()

                    isInserted=False
                    for attendance in AttendanceModel.objects.filter(studentid=Id):
                        print("1",attendance.date,"\t"+attendance.intime)
                        print("2",cd,"\t",str(cr))
                        print("22",type(str(attendance.date)),type(cd), "\t",type(attendance.intime),type(cr))
                        if str(attendance.date)==cd and str(attendance.intime)==str(cr):
                            print("3","in if")
                            isInserted=True

                    print("4",isInserted)
                    if not isInserted:
                        AttendanceModel(studentid=Id, date=cd, intime=cr, outtime=0,branch=student.branch,isattended="no").save()

                    cv2.putText(im,str(Id), (x + h, y), font, 1, (255, 255, 0,),2)

            cv2.imshow(str(Id), im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cam.release()
        cv2.destroyAllWindows()

    return

def attendanceOut():

    now = time.time()
    future = now + 20

    if time.time() < future:
        recognizer = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer()
        try:
            recognizer.read(constants.dataset_path+"TrainingImageLabel\\Trainner.yml")
        except Exception as e:
            print(e)

        harcascadePath = constants.path+"haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(harcascadePath)
        cam = cv2.VideoCapture(0)
        time.sleep(2.0)
        font = cv2.FONT_HERSHEY_SIMPLEX

        while True:
            ret, im = cam.read()
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.2, 5)
            for (x, y, w, h) in faces:
                Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
                if (conf < 70):
                    print(conf)
                    cv2.rectangle(im, (x, y), (x + w, y + h), (0, 260, 0), 7)

                    td = datetime.now()
                    cd=str(td.day)+"-"+str(td.month)+"-"+str(td.year)
                    cr=td.hour

                    isInserted=False
                    for attendance in AttendanceModel.objects.filter(studentid=Id):
                        if str(attendance.date)==cd and str(attendance.outtime)==str(cr):
                            isInserted=True

                    if not isInserted:
                        att=AttendanceModel.objects.filter(studentid=Id).first()
                        if att is not None:
                            AttendanceModel.objects.filter(studentid=Id).update(outtime=cr,isattended="yes")

                    cv2.putText(im,str(Id), (x + h, y), font, 1, (255, 255, 0,),2)

            cv2.imshow(str(Id), im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cam.release()
        cv2.destroyAllWindows()

    return