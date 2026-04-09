from flask import Flask, request, jsonify
import cv2
import numpy as np
import pytesseract
import re
from PIL import Image
import io
import os
import shutil
import signal
import threading

# Try to load pyzbar, fallback to OpenCV-only if DLLs missing
try:
    from pyzbar.pyzbar import decode as pyzbar_decode
    HAS_PYZBAR = True
    print("[OK] pyzbar loaded successfully")
except (ImportError, FileNotFoundError):
    HAS_PYZBAR = False
    print("[WARN] pyzbar not available, using OpenCV QR detector only")

app = Flask(__name__)

# Auto-detect Tesseract path on Windows
if os.name == 'nt':
    tesseract_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
    ]
    found = shutil.which('tesseract')
    if found:
        pytesseract.pytesseract.tesseract_cmd = found
    else:
        for p in tesseract_paths:
            if os.path.isfile(p):
                pytesseract.pytesseract.tesseract_cmd = p
                break

@app.route('/qr', methods=['POST'])
def decode_qr_and_text():
    try:
        file = request.files['image']
        img_bytes = file.read()
        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        # Resize ภาพใหญ่ลงก่อนทุกอย่างเพื่อประหยัด memory
        h, w = img.shape[:2]
        max_dim = 1500
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            print(f"Resized image from {w}x{h} to {img.shape[1]}x{img.shape[0]}")

        # 1. ถอดรหัส QR Code
        qr_data = decode_qr_code(img)

        # 2. อ่านข้อความจากภาพ (OCR) - ทำเสมอเพื่อดึงยอดเงิน/บัญชี
        # ใช้ timeout เพื่อกัน OCR ค้างบน free tier
        ocr_text = ""
        ocr_result = [None]

        def run_ocr():
            try:
                ocr_result[0] = extract_text_from_image(img)
            except Exception as e:
                print(f"OCR thread error: {e}")

        ocr_thread = threading.Thread(target=run_ocr)
        ocr_thread.start()
        ocr_thread.join(timeout=30)  # รอ OCR สูงสุด 30 วินาที

        if ocr_thread.is_alive():
            print("⏰ OCR timeout (30s) — ใช้ QR data อย่างเดียว")
        elif ocr_result[0]:
            ocr_text = ocr_result[0]
            print(f"OCR result length: {len(ocr_text)}")

        # 3. วิเคราะห์ข้อมูลจากข้อความ
        slip_data = extract_slip_info(ocr_text)
        
        # 4. รวมข้อมูล QR และ OCR
        result = {
            "qr_code": qr_data if qr_data else None,
            "amount": slip_data.get("amount"),
            "sender_name": slip_data.get("sender_name"),
            "receiver_name": slip_data.get("receiver_name"),
            "receiver_account": slip_data.get("receiver_account"),
            "transaction_time": slip_data.get("transaction_time"),
            "bank_name": slip_data.get("bank_name"),
            "reference": slip_data.get("reference"),
            "raw_text": ocr_text[:500] if ocr_text else None
        }

        if qr_data or any(v for k, v in result.items() if k != 'raw_text'):
            return jsonify(result)
        else:
            return jsonify({"error": "No QR code or text found"}), 400

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


def extract_text_from_image(img):
    """
    ใช้ Tesseract OCR อ่านข้อความจากภาพ
    ปรับให้เร็วสำหรับ free tier cloud
    """
    # Resize ภาพให้เล็กมาก เพื่อให้ OCR เร็ว
    h, w = img.shape[:2]
    max_dim = 800
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        print(f"OCR resize to {img.shape[1]}x{img.shape[0]}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # วิธีเดียวที่เร็วสุด: grayscale + contrast, OEM 1 (LSTM only = เร็วกว่า)
    try:
        enhanced = cv2.convertScaleAbs(gray, alpha=1.3, beta=20)
        text = pytesseract.image_to_string(enhanced, lang='tha+eng', config='--psm 6 --oem 1')
        if text.strip():
            return text.strip()
    except Exception as e:
        print(f"OCR error: {e}")

    return ""


def extract_slip_info(text):
    """
    วิเคราะห์ข้อมูลจากข้อความสลิป
    """
    result = {}
    
    if not text:
        return result
    
    # ลบบรรทัดว่างและทำความสะอาดข้อความ
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    full_text = ' '.join(lines)
    
    # ดึงยอดเงิน
    amount_patterns = [
        r'(?:amount|Amount|ยอดเงิน|จำนวนเงิน|โอนเงิน)[:\s]*([0-9,]+\.?\d*)',
        r'([0-9,]+\.?\d*)\s*(?:บาท|THB|Baht)',
        r'(?:Total|รวม)[:\s]*([0-9,]+\.?\d*)',
        r'([0-9]{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:บาท|THB)',
        r'THB\s*([0-9,]+\.?\d*)'
    ]
    
    for pattern in amount_patterns:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            amount = match.group(1).replace(',', '')
            try:
                result["amount"] = f"{float(amount):,.2f}"
                break
            except:
                continue
    
    # ดึงชื่อผู้รับ
    receiver_patterns = [
        r'(?:To|ไปยัง|ผู้รับ|Receiver|ชื่อผู้รับ|โอนให้)[:\s]*([^\n\r]+)',
        r'(?:รับโอน|Transfer to)[:\s]*([^\n\r]+)',
        r'(?:บัญชี)[:\s]*([^\n\r]+?)(?:\s*[0-9Xx\-]+|$)'
    ]
    
    for pattern in receiver_patterns:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            name = re.sub(r'\s*[0-9Xx\-]{5,}.*$', '', name).strip()
            if name and len(name) > 2:
                result["receiver_name"] = name
                break
    
    # ดึงชื่อผู้โอน
    sender_patterns = [
        r'(?:From|จาก|ผู้โอน|Sender|ชื่อผู้ส่ง)[:\s]*([^\n\r]+)',
        r'(?:โอนจาก|Transfer from)[:\s]*([^\n\r]+)',
        r'(?:บัญชีต้นทาง)[:\s]*([^\n\r]+)'
    ]
    
    for pattern in sender_patterns:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            name = re.sub(r'\s*[0-9Xx\-]{5,}.*$', '', name).strip()
            if name and len(name) > 2:
                result["sender_name"] = name
                break
    
    # ดึงหมายเลขบัญชี
    account_patterns = [
        r'(?:Account No|เลขที่บัญชี|บัญชี|เลขบัญชี)[:\s]*([0-9\-Xx*]+)',
        r'([0-9]{3}-[0-9]-[0-9]{5}-[0-9])',
        r'([Xx*0-9]{3,}[-\s]?[Xx*0-9]{3,}[-\s]?[0-9]{3,})'
    ]
    
    for pattern in account_patterns:
        match = re.search(pattern, full_text)
        if match:
            result["receiver_account"] = match.group(1).strip()
            break
    
    # ดึงชื่อธนาคาร
    bank_patterns = [
        r'(กรุงเทพ|กสิกรไทย|ไทยพาณิชย์|กรุงไทย|ทหารไทย|ออมสิน|อาคารสงเคราะห์|กรุงศรี|ยูโอบี|ซีไอเอ็มบี|ทีทีบี|แลนด์|เกียรตินาคิน)',
        r'(Bangkok Bank|Kasikorn|KBANK|SCB|Krung Thai|KTB|TMB|TTB|GSB|GHB|BAY|UOB|CIMB|LH Bank)',
        r'(K\s*PLUS|SCB\s*EASY|Krungthai\s*NEXT|ttb\s*touch)',
        r'(ธนาคาร[^\s\n]+)'
    ]
    
    for pattern in bank_patterns:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            result["bank_name"] = match.group(1).strip()
            break
    
    # ดึงเวลาทำรายการ
    time_patterns = [
        r'(\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}[:.]\d{2}(?:[:.]\d{2})?)',
        r'(\d{1,2}-\d{1,2}-\d{2,4}\s+\d{1,2}[:.]\d{2})',
        r'(\d{4}-\d{1,2}-\d{1,2}\s+\d{1,2}[:.]\d{2})',
        r'(\d{1,2}\s+(?:ม\.ค\.|ก\.พ\.|มี\.ค\.|เม\.ย\.|พ\.ค\.|มิ\.ย\.|ก\.ค\.|ส\.ค\.|ก\.ย\.|ต\.ค\.|พ\.ย\.|ธ\.ค\.)\s+\d{2,4}\s+\d{1,2}[:.]\d{2})'
    ]
    
    for pattern in time_patterns:
        match = re.search(pattern, full_text)
        if match:
            result["transaction_time"] = match.group(1).strip()
            break
    
    # ดึงหมายเลขอ้างอิง
    ref_patterns = [
        r'(?:Ref\.?\s*No\.?|Reference|อ้างอิง|เลขที่รายการ)[:\s]*([A-Za-z0-9]+)',
        r'(?:Transaction\s*ID|รายการ)[:\s]*([A-Za-z0-9]+)',
        r'(?:หมายเลขอ้างอิง)[:\s]*([A-Za-z0-9]+)'
    ]
    
    for pattern in ref_patterns:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            result["reference"] = match.group(1).strip()
            break
    
    return result


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "ok"})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Slip Verification API on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=True)
