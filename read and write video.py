import cv2
cap=cv2.VideoCapture(r"D:\CVPR\cvpr lab\5d0\5d0\input_video.mp4")
if not cap.isOpened():
    print("Error")
    exit()
output=cv2.VideoWriter(r"D:\CVPR\cvpr lab\output2.avi",cv2.VideoWriter_fourcc(*'XVID'),10,(500,500))
while True:
    ret,frame=cap.read()
    if not ret:
        print("Error")
        break
    frame=cv2.resize(frame,(500,500))
    cv2.rectangle(frame,(50,50),(150,150),(0,255,0),4)
    cv2.line(frame,(50,50),(150,150),(0,255,0),3)
    cv2.circle(frame,(250,250),50,(0,255,0),2)
    output.write(frame)
    cv2.imshow("output",frame)
    if cv2.waitKey(1) & 0xFF==ord("s"):
        break
cv2.destroyAllWindows()
cap.release()
output.release()
