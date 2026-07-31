# import cv2

# camera = cv2.VideoCapture(0)

# while True:

#     success, frame = camera.read()

#     if not success:
#         break

#     cv2.imshow(
#         "NAUB Surveillance System Camera",
#         frame
#     )

#     if cv2.waitKey(1) == 27:
#         break

# camera.release()
# cv2.destroyAllWindows()
import cv2

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Cannot read camera")
        break

    cv2.imshow("NAUB Surveillance System Camera", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()