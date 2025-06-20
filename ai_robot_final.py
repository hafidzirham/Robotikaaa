# KODE FINAL UNTUK PC/LAPTOP (OTAK AI HYBRID v4 - LOGIKA BELOK DITINGKATKAN)

import cv2
import serial
import time
import torch
import numpy as np
from ultralytics import YOLO

# --- PENGATURAN WAJIB ---
# 1. Pastikan port serial ESP32 Anda sudah benar.
SERIAL_PORT = 'COM5' 
# 2. Objek yang menjadi tugas utama untuk diikuti.
PRIMARY_TARGET = 'cell phone'
# 3. Objek yang menjadi prioritas utama untuk dihindari.
AVOID_TARGET = 'chair'
# 4. Threshold jarak (Nilai LEBIH TINGGI berarti LEBIH DEKAT).
DISTANCE_THRESHOLD = 5.0 
# 5. Waktu tunggu sebelum mundur jika terjebak (dalam detik).
REVERSE_WAIT_TIME = 2.0 
# -------------------------

try:
    ser = serial.Serial(SERIAL_PORT, 115200, timeout=1)
    print(f"✅ Berhasil terhubung ke ESP32 di port {SERIAL_PORT}")
    time.sleep(2)
except serial.SerialException as e:
    print(f"❌ Error: Gagal terhubung ke port {SERIAL_PORT}.")
    print("   Pastikan port benar dan tidak digunakan program lain (misal: Serial Monitor).")
    exit()

print("🧠 Memuat model AI... (Butuh internet saat pertama kali)")
try:
    model_yolo = YOLO('yolov8n.pt')
    model_midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_midas.to(device)
    model_midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform
    print("✅ Model AI berhasil dimuat.")
except Exception as e:
    print(f"❌ Gagal memuat model. Error: {e}")
    exit()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Tidak bisa membuka webcam.")
    ser.close()
    exit()

print(f"\n🚀 Sistem AI Aktif. Ikuti '{PRIMARY_TARGET}', Hindari '{AVOID_TARGET}'.")
last_command, is_stuck, stuck_timestamp = '', False, 0

while True:
    ret, frame = cap.read()
    if not ret: break

    frame_height, frame_width, _ = frame.shape

    # Analisis Jarak & Objek
    img_transformed = transform(frame).to(device)
    with torch.no_grad():
        prediction = model_midas(img_transformed)
        prediction = torch.nn.functional.interpolate(prediction.unsqueeze(1), size=frame.shape[:2], mode="bicubic", align_corners=False).squeeze()
    depth_map = prediction.cpu().numpy()
    results_yolo = model_yolo(frame, verbose=False)
    annotated_frame = results_yolo[0].plot()

    avoid_target_spotted, primary_target_spotted = False, False
    target_x_center, target_avg_depth = 0, 0

    for r in results_yolo:
        for box in r.boxes:
            class_name = model_yolo.names[int(box.cls[0])]
            if class_name == AVOID_TARGET:
                avoid_target_spotted = True
            elif class_name == PRIMARY_TARGET:
                primary_target_spotted = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                target_x_center = int((x1 + x2) / 2)
                roi_depth = depth_map[y1:y2, x1:x2]
                target_avg_depth = np.mean(roi_depth) if roi_depth.size > 0 else 0
                cv2.putText(annotated_frame, f"Jarak: {target_avg_depth:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    command = '0'
    if avoid_target_spotted:
        command, is_stuck = '0', False
        cv2.putText(annotated_frame, f"STOP: {AVOID_TARGET}!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    elif primary_target_spotted:
        # --- LOGIKA BARU UNTUK BELOK YANG LEBIH TEGAS ---
        # Prioritas pertama: hadapkan badan robot ke target
        if target_x_center < frame_width / 3:
            command = '3' # Belok Kiri
            is_stuck = False # Reset status terjebak saat berputar
        elif target_x_center > frame_width * 2 / 3:
            command = '2' # Belok Kanan
            is_stuck = False # Reset status terjebak saat berputar
        # Jika target sudah di tengah, barulah cek jarak dan maju/mundur
        else:
            if target_avg_depth > DISTANCE_THRESHOLD:
                if not is_stuck:
                    is_stuck, stuck_timestamp, command = True, time.time(), '0'
                else:
                    if time.time() - stuck_timestamp > REVERSE_WAIT_TIME:
                        command = '4'
                    else:
                        command = '0'
            else:
                is_stuck = False
                command = '1' # Maju
    else:
        is_stuck = False

    if command != last_command:
        ser.write(command.encode('utf-8'))
        print(f"Prioritas: Hindari='{avoid_target_spotted}', Ikuti='{primary_target_spotted}' -> Perintah: {command}")
        last_command = command

    cv2.imshow("Otak AI Robot - Tekan 'q' untuk keluar", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

print("Mematikan sistem...")
ser.write(b'0')
ser.close()
cap.release()
cv2.destroyAllWindows()
print("Sistem berhasil dimatikan.")
