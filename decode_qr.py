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
            # ตรวจสอบความถูกต้องของ QR ตามมาตรฐาน EMVCo/PromptPay
            validation = validate_emvco_qr(qr_data)
            return jsonify({
                "qr_code": qr_data,
                "qr_valid": validation["valid"],
                "qr_info": validation["info"],
                "qr_warnings": validation["warnings"]
            })
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


def parse_emvco_tlv(data):
    """แกะโครงสร้าง TLV (Tag-Length-Value) ของ QR Code"""
    result = {}
    i = 0
    while i + 4 <= len(data):
        tag = data[i:i+2]
        try:
            length = int(data[i+2:i+4])
        except ValueError:
            break
        if i + 4 + length > len(data):
            break
        value = data[i+4:i+4+length]
        result[tag] = value
        i += 4 + length
    return result


def crc16_ccitt(data):
    """คำนวณ CRC-16/CCITT-FALSE"""
    crc = 0xFFFF
    for byte in data.encode('utf-8') if isinstance(data, str) else data:
        if isinstance(byte, str):
            byte = ord(byte)
        crc ^= (byte << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc = crc << 1
            crc &= 0xFFFF
    return crc


def validate_emvco_qr(qr_data):
    """
    ตรวจความถูกต้องของ QR Code จากสลิปธนาคารไทย
    สลิปไทยใช้ TLV format: Tag 00 (ข้อมูลธุรกรรม), Tag 51 (ประเทศ), Tag 91 (checksum)

    จับสลิปปลอม:
    - QR โครงสร้างผิด / ไม่ใช่ TLV
    - ไม่มี Tag ที่จำเป็น (51=TH, 91=checksum)
    - ข้อมูลภายในไม่สมเหตุสมผล
    """
    result = {"valid": False, "info": {}, "warnings": []}

    if not qr_data or len(qr_data) < 20:
        result["warnings"].append("QR สั้นเกินไป อาจเป็นของปลอม")
        return result

    if len(qr_data) > 300:
        result["warnings"].append("QR ยาวผิดปกติ")
        return result

    # 1. แกะ TLV
    tags = parse_emvco_tlv(qr_data)

    if not tags or len(tags) < 2:
        result["warnings"].append("แกะโครงสร้าง QR ไม่ได้ (ไม่ใช่ TLV)")
        return result

    result["info"]["tags_found"] = list(tags.keys())

    # 2. ตรวจ Tag 00 (ข้อมูลธุรกรรมหลัก)
    if "00" in tags:
        tag00 = tags["00"]
        result["info"]["data_length"] = len(tag00)

        # ข้อมูลในสลิปจริงขึ้นต้นด้วย "0006" (format header)
        if tag00.startswith("0006"):
            result["info"]["format"] = "Thai Bank Slip"
        elif tag00 == "01":
            result["info"]["format"] = "EMVCo Payment QR"
        else:
            result["info"]["format"] = "unknown"
            result["warnings"].append("รูปแบบข้อมูล Tag 00 ไม่ตรงกับสลิปธนาคารไทย")

        # ดึง reference จาก tag00 (ตัวอักษรและตัวเลข)
        ref_match = re.search(r'[A-Z]{2,}[0-9]{3,}', tag00)
        if ref_match:
            result["info"]["reference"] = ref_match.group()
    else:
        result["warnings"].append("ไม่พบ Tag 00 (ข้อมูลหลักของ QR)")

    # 3. ตรวจ Country Tag 51 = "TH"
    if "51" in tags:
        result["info"]["country"] = tags["51"]
        if tags["51"] != "TH":
            result["warnings"].append("ประเทศไม่ใช่ TH: " + tags["51"])
    elif "58" in tags:
        # บาง QR ใช้ Tag 58 แทน
        result["info"]["country"] = tags["58"]
        if tags["58"] != "TH":
            result["warnings"].append("ประเทศไม่ใช่ TH: " + tags["58"])
    else:
        result["warnings"].append("ไม่พบรหัสประเทศ (Tag 51/58)")

    # 4. ตรวจ Checksum Tag 91
    if "91" in tags:
        result["info"]["checksum"] = tags["91"]
        # checksum ควรเป็น hex 4 ตัว
        if not re.match(r'^[0-9A-Fa-f]{4}$', tags["91"]):
            result["warnings"].append("Checksum รูปแบบผิดปกติ: " + tags["91"])
    else:
        result["warnings"].append("ไม่พบ Checksum (Tag 91)")

    # 5. ตรวจ EMVCo Payment QR (กรณีเป็น PromptPay QR)
    if "53" in tags:
        result["info"]["currency"] = tags["53"]
    if "54" in tags:
        try:
            result["info"]["amount"] = float(tags["54"])
        except ValueError:
            pass
    if "63" in tags:
        # CRC-16 validation สำหรับ EMVCo standard QR
        crc_in_qr = tags["63"].upper()
        crc_pos = qr_data.rfind("6304")
        if crc_pos >= 0:
            crc_payload = qr_data[:crc_pos + 4]
            calculated = format(crc16_ccitt(crc_payload), '04X')
            result["info"]["crc_match"] = (crc_in_qr == calculated)
            if crc_in_qr != calculated:
                result["warnings"].append("CRC ไม่ตรง! QR อาจถูกแก้ไข")

    # 6. ตรวจ reconstructed length ตรงกับ QR จริง
    reconstructed_len = sum(4 + len(v) for v in tags.values())
    if reconstructed_len != len(qr_data):
        result["warnings"].append("ความยาว QR ไม่ตรงกับโครงสร้าง TLV (เกิน " + str(len(qr_data) - reconstructed_len) + " ตัวอักษร)")

    # สรุป: valid ถ้ามี tag หลักครบและไม่มี warning ร้ายแรง
    has_country = ("51" in tags and tags.get("51") == "TH") or ("58" in tags and tags.get("58") == "TH")
    has_checksum = "91" in tags or "63" in tags
    has_data = "00" in tags
    no_critical = all("CRC ไม่ตรง" not in w and "แกะโครงสร้าง" not in w and "สั้นเกินไป" not in w for w in result["warnings"])

    result["valid"] = has_data and has_country and has_checksum and no_critical

    return result


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "ok"})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Slip Verification API on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=True)
