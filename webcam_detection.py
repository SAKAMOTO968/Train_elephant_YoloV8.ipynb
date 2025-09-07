import cv2
from ultralytics import YOLO

# โหลดโมเดลที่ผ่านการเทรนแล้ว
# ตรวจสอบให้แน่ใจว่าไฟล์ best.pt อยู่ในโฟลเดอร์เดียวกัน
model = YOLO(r'C:\Users\month\Desktop\WORK\Elephant_detection\best.pt')

# เปิดการใช้งานกล้องเว็บแคม (0 หมายถึงกล้องเริ่มต้น)
cap = cv2.VideoCapture(0)

# ตรวจสอบว่ากล้องเปิดสำเร็จหรือไม่
if not cap.isOpened():
    print("ไม่สามารถเปิดกล้องเว็บแคมได้ โปรดตรวจสอบการเชื่อมต่อ")
    exit()

print("กำลังเริ่มการตรวจจับ... กด 'q' เพื่อหยุด")

# วนลูปเพื่ออ่านภาพจากกล้องแบบเรียลไทม์
while True:
    # อ่านเฟรมจากกล้อง
    ret, frame = cap.read()
    if not ret:
        break

    # ใช้โมเดลเพื่อทำการตรวจจับในเฟรมปัจจุบัน
    # 'stream=True' ทำให้การประมวลผลเร็วขึ้น
    results = model(frame, stream=True)

    # วนลูปเพื่อแสดงผลลัพธ์การตรวจจับ
    for r in results:
        # ใช้ฟังก์ชัน .plot() เพื่อวาดกรอบ, ชื่อคลาส และคะแนนความมั่นใจ
        annotated_frame = r.plot()
        
    # แสดงเฟรมที่มีการตรวจจับแล้วบนหน้าต่าง
    cv2.imshow("Real-time Detection", annotated_frame)

    # กด 'q' เพื่อออกจากโปรแกรม
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดการทำงานของกล้องและหน้าต่างแสดงผล
cap.release()
cv2.destroyAllWindows()