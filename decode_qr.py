from flask import Flask, request, jsonify
import cv2
import numpy as np
import re
import os
import shutil

# Try to load pyzbar, fallback to OpenCV-only if DLLs missing
try:
    from pyzbar.pyzbar import decode as pyzbar_decode
    HAS_PYZBAR = True
    print("[OK] pyzbar loaded successfully")
except (ImportError, FileNotFoundError):
    HAS_PYZBAR = False
    print("[WARN] pyzbar not available, using OpenCV QR detector only")

app = Flask(__name__)


@app.route('/qr', methods=['POST'])
def decode_qr_and_text():
    try:
        file = request.files['image']
        img_bytes = file.read()
        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        # Resize ภาพใหญ่ลงก่อนเพื่อประหยัด memory
        h, w = img.shape[:2]
        max_dim = 1500
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        # ถอดรหัส QR Code
        qr_data = decode_qr_code(img)

        if qr_data:
            return jsonify({"qr_code": qr_data})
        else:
            return jsonify({"error": "No QR code found"}), 400

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": f"Error: {str(e)}"}), 500


def decode_qr_code(img):
    """
    อ่าน QR Code ด้วย pyzbar (หลัก) + OpenCV (สำรอง)
    ลดวิธีเหลือ 2 เพื่อความเร็ว
    """
    # === pyzbar (if available) ===
    if HAS_PYZBAR:
        # 1. ลอง pyzbar กับภาพต้นฉบับ
        try:
            results = pyzbar_decode(img)
            if results:
                qr_data = results[0].data.decode('utf-8')
                print(f"[OK] pyzbar decoded: {qr_data[:80]}...")
                return qr_data
        except Exception as e:
            print(f"pyzbar error (original): {e}")

        # 2. ลอง pyzbar กับภาพ grayscale + contrast
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)
            results = pyzbar_decode(enhanced)
            if results:
                qr_data = results[0].data.decode('utf-8')
                print(f"[OK] pyzbar decoded (enhanced): {qr_data[:80]}...")
                return qr_data
        except Exception as e:
            print(f"pyzbar error (enhanced): {e}")

    # === OpenCV QRCodeDetector fallback ===
    try:
        qr = cv2.QRCodeDetector()
        qr_data, _, _ = qr.detectAndDecode(img)
        if qr_data:
            print(f"[OK] OpenCV QR decoded: {qr_data[:80]}...")
            return qr_data
    except Exception as e:
        print(f"OpenCV QR error: {e}")

    return None


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "ok"})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Slip Verification API on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=True)
