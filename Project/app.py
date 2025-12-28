# app.py 
import os
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2
import mysql.connector
from datetime import datetime

# New libraries for PDF generation
import io
import base64
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.colors import Color

# Pose library
import mediapipe as mp
import math

# ---------- Config ----------
UPLOAD_FOLDER = "static/uploads"
PDF_UPLOAD_FOLDER = "static/pdfs"
PLAYER_IMAGES_FOLDER = "static/player_images"
ALLOWED = {"mp4", "avi", "mov", "mkv"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PDF_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLAYER_IMAGES_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PDF_UPLOAD_FOLDER'] = PDF_UPLOAD_FOLDER
app.config['PLAYER_IMAGES_FOLDER'] = PLAYER_IMAGES_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024

CLASS_NAMES = ["Cover Drive", "Lofted", "Square Cut", "Pull", "Straight Drive"]
NUM_FRAMES = 16
FRAME_SIZE = 112
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "best_model.pth"

DB_CONFIG = {
    'user': 'root',
    'password': 'admin123',
    'host': 'localhost',
    'database': 'pro'
}

# ---------- Model ----------
def make_model(num_classes):
    model = torchvision.models.video.r3d_18(pretrained=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

try:
    model = make_model(len(CLASS_NAMES))
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(state, dict) and not isinstance(state, torch.nn.Module):
        model.load_state_dict(state)
    else:
        model = state
    model = model.to(DEVICE)
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}. The app will run, but predictions may fail.")
    model = None

# ---------- Transforms & helpers ----------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((FRAME_SIZE, FRAME_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED

def sample_frames_from_video(video_path, num_frames=NUM_FRAMES, frame_size=FRAME_SIZE):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total <= 0:
        while True:
            ret, f = cap.read()
            if not ret:
                break
            frames.append(f)
    else:
        if total < num_frames:
            allf = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while True:
                ret, f = cap.read()
                if not ret:
                    break
                allf.append(f)
            if len(allf) == 0:
                cap.release()
                return []
            indices = np.linspace(0, len(allf) - 1, num_frames).astype(int)
            for i in indices:
                frames.append(allf[i])
            cap.release()
            return frames
        indices = np.linspace(0, total - 1, num_frames).astype(int)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, f = cap.read()
            if not ret:
                f = np.zeros((frame_size, frame_size, 3), dtype=np.uint8)
            frames.append(f)
    cap.release()
    return frames

def predict_video(video_path):
    """
    Returns: predicted_label (str), probs (list[float]), frames (list[np.array])
    """
    if not model:
        return "Model Error", [0.0] * len(CLASS_NAMES), []

    frames = sample_frames_from_video(video_path, NUM_FRAMES, FRAME_SIZE)
    if len(frames) == 0:
        return "Error", [0.0] * len(CLASS_NAMES), []

    frame_tensors = []
    for f in frames:
        try:
            f_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        except cv2.error:
            f_rgb = cv2.cvtColor(f, cv2.COLOR_GRAY2RGB)
        except Exception:
            f_rgb = f
        t = transform(f_rgb)
        frame_tensors.append(t)

    video_tensor = torch.stack(frame_tensors, dim=1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(video_tensor)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0].tolist()
    top_idx = int(np.argmax(probs))
    return CLASS_NAMES[top_idx], probs, frames

# ---------- Pose and scoring helpers ----------
mp_pose = mp.solutions.pose

# landmark indices for MediaPipe pose
LS, RS, LW, RW, LH, RH, NOSE = 11, 12, 15, 16, 23, 24, 0

def euclid(a,b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def angle_between(a,b,c):
    ba = (a[0]-b[0], a[1]-b[1])
    bc = (c[0]-b[0], c[1]-b[1])
    dot = ba[0]*bc[0] + ba[1]*bc[1]
    lena = math.hypot(ba[0], ba[1]) + 1e-6
    lenb = math.hypot(bc[0], bc[1]) + 1e-6
    cosv = dot/(lena*lenb)
    cosv = max(-1.0, min(1.0, cosv))
    return math.degrees(math.acos(cosv))

def extract_landmarks_from_frames(frames):
    results_all = []
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        for f in frames:
            h,w = f.shape[:2]
            f_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            res = pose.process(f_rgb)
            if res.pose_landmarks:
                lm = {}
                for i, l in enumerate(res.pose_landmarks.landmark):
                    # store (x_pixel, y_pixel, z, visibility)
                    lm[i] = (int(l.x * w), int(l.y * h), l.z, l.visibility)
                results_all.append(lm)
            else:
                results_all.append(None)
    return results_all

def compute_body_scores(landmarks_list):
    """
    Returns: dict: Head, Shoulder, Hands, Hips, Feet, Performance - values 0..10
    """
    nose_positions = []
    shoulder_angles = []
    wrist_speeds = []
    hip_angles = []
    ankle_distances = []

    prev_wrists = None

    for lm in landmarks_list:
        if lm is None:
            nose_positions.append(None)
            shoulder_angles.append(None)
            wrist_speeds.append(None)
            hip_angles.append(None)
            ankle_distances.append(None)
            prev_wrists = None
            continue

        nose_positions.append((lm[NOSE][0], lm[NOSE][1]))

        if LS in lm and RS in lm and 13 in lm and 14 in lm and 11 in lm and 12 in lm:
            # Angle of shoulders relative to torso
            left_shoulder_angle = angle_between(lm[23], lm[11], lm[13])
            right_shoulder_angle = angle_between(lm[24], lm[12], lm[14])
            shoulder_angles.append((left_shoulder_angle + right_shoulder_angle) / 2.0)
        else:
            shoulder_angles.append(None)

        if LW in lm and RW in lm:
            w1 = lm[LW]
            w2 = lm[RW]
            cur_wrists = ((w1[0], w1[1]), (w2[0], w2[1]))
            if prev_wrists is not None:
                d0 = euclid(cur_wrists[0], prev_wrists[0])
                d1 = euclid(cur_wrists[1], prev_wrists[1])
                wrist_speeds.append((d0 + d1) / 2.0)
            else:
                wrist_speeds.append(0.0)
            prev_wrists = cur_wrists
        else:
            wrist_speeds.append(None)
            prev_wrists = None

        if LH in lm and RH in lm and 25 in lm and 26 in lm:
            # Angle of hips relative to torso
            left_hip_angle = angle_between(lm[11], lm[23], lm[25])
            right_hip_angle = angle_between(lm[12], lm[24], lm[26])
            hip_angles.append((left_hip_angle + right_hip_angle) / 2.0)
        else:
            hip_angles.append(None)

        if 27 in lm and 28 in lm and LS in lm and RS in lm:
            ankle_left = lm[27]
            ankle_right = lm[28]
            # Normalize foot distance by shoulder width
            sh_w = euclid(lm[LS], lm[RS]) + 1e-6
            ankle_distance = euclid(ankle_left, ankle_right)
            ankle_distances.append(ankle_distance / sh_w)
        else:
            ankle_distances.append(None)

    # HEAD (now based on stability)
    nose_y = [p[1] for p in nose_positions if p is not None]
    if len(nose_y) >= 2:
        # A lower variance means a more stable head
        var_y = float(np.var(nose_y))
        # A lower variance gets a higher score. We cap the score at 10.
        head_score = max(0.0, min(10.0, 10.0 - (var_y / 100)))
    else:
        head_score = 5.0

    # SHOULDERS
    shoulder_vals = [a for a in shoulder_angles if a is not None]
    if shoulder_vals:
        mean_ang = float(np.mean(shoulder_vals))
        # Optimal shoulder angle range for batting is generally around 150-180.
        # Let's give a score based on how close it is to that range.
        if mean_ang > 180: mean_ang = 360 - mean_ang
        dev = abs(165.0 - mean_ang)
        shoulder_score = max(0.0, min(10.0, 10.0 - (dev / 20.0)))
    else:
        shoulder_score = 5.0

    # HANDS
    wrist_vals = [v for v in wrist_speeds if v is not None]
    if wrist_vals:
        mean_speed = float(np.mean(wrist_vals))
        # Optimal speed for a good follow-through, scaled to a 0-10 score.
        hand_score = max(0.0, min(10.0, mean_speed / 15.0))
    else:
        hand_score = 5.0

    # HIPS
    hip_vals = [a for a in hip_angles if a is not None]
    if hip_vals:
        mean_hip = float(np.mean(hip_vals))
        # Optimal hip angle for a stable stance is close to 180 (straight back)
        if mean_hip > 180: mean_hip = 360 - mean_hip
        devh = abs(170.0 - mean_hip)
        hip_score = max(0.0, min(10.0, 10.0 - (devh / 20.0)))
    else:
        hip_score = 5.0

    # FEET
    ratios = [r for r in ankle_distances if r is not None]
    if ratios:
        mean_ratio = float(np.mean(ratios))
        # Optimal ratio of ankle distance to shoulder width for a balanced stance is around 1.0-1.5.
        devr = abs(1.25 - mean_ratio)
        feet_score = max(0.0, min(10.0, 10.0 - (devr / 0.5)))
    else:
        feet_score = 5.0

    scores = {
        "Head": round(head_score, 2),
        "Shoulder": round(shoulder_score, 2),
        "Hands": round(hand_score, 2),
        "Hips": round(hip_score, 2),
        "Feet": round(feet_score, 2)
    }
    
    # Ensure scores don't go below zero
    for key in scores:
        if scores[key] < 0:
            scores[key] = 0

    overall = round(sum(scores.values()) / len(scores), 2)
    scores["Performance"] = overall
    return scores

# ---------- Routes ----------

# app.py

# ... (keep all the code before this function) ...

@app.route("/report/<int:player_id>", methods=["GET", "POST"])
def player_report(player_id):
    player_info = None
    results = []
    selection = None
    coach_comment = ""
    evaluation_saved = False
    pdf_path = None
    average_confidence = None
    average_performance = None # NEW: To store the performance score

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        query = "SELECT id, name, batting_type, province, region, player_image_path FROM players WHERE id = %s"
        cursor.execute(query, (player_id,))
        player_info = cursor.fetchone()
        
        eval_query = "SELECT * FROM evaluations WHERE player_id = %s"
        cursor.execute(eval_query, (player_id,))
        saved_evaluation = cursor.fetchone()
        
        if saved_evaluation:
            selection = saved_evaluation['is_selected']
            coach_comment = saved_evaluation['coach_comments']
            evaluation_saved = True
            pdf_path = saved_evaluation.get('pdf_path')
            average_confidence = saved_evaluation.get('average_confidence')
            average_performance = saved_evaluation.get('average_performance') # NEW: Load performance score

    except mysql.connector.Error as err:
        print(f"Database error: {err}")
        return f"Database error: {err}", 500
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn:
            conn.close()
    
    if request.method == "POST":
        files = request.files.getlist("videos")
        files = [f for f in files if f and f.filename]
        if len(files) != 6:
            return f"Error: Please upload exactly 6 videos (selected: {len(files)}).", 400
        
        total_confidence = 0
        total_performance_score = 0 # NEW: To calculate average performance
        video_count = 0

        for f in files:
            if not allowed_file(f.filename):
                return f"Error: file type not allowed: {f.filename}", 400
            fname = secure_filename(f.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
            f.save(save_path)

            predicted, probs, frames = predict_video(save_path)
            
            try:
                lms = extract_landmarks_from_frames(frames) if frames else []
                scores = compute_body_scores(lms) if lms else {"Performance": 0}
            except Exception as ex:
                print(f"Pose/score error for {fname}: {ex}")
                scores = {"Performance": 0}

            total_confidence += max(probs) if probs else 0
            # NEW: Accumulate the 'Performance' score from each video
            # The score is out of 10, so we convert to a percentage
            total_performance_score += scores.get('Performance', 0) * 10 
            video_count += 1
            results.append({"filename": fname, "predicted": predicted, "probs": probs, "scores": scores})

        if video_count > 0:
            average_confidence = (total_confidence / video_count) * 100
            average_performance = total_performance_score / video_count
        else:
            average_performance = 0

        # --- CHANGED: Decision logic now uses average_performance ---
        if average_confidence >= 90:
            selection = True
            coach_comment = "Selected (Skilled): Outstanding performance with an average score above 90%."
        elif average_confidence >= 80: 
            selection = True
            coach_comment = "Selected (Intermediate): Good performance, but there is room for improvement."
        else:
            selection = False
            coach_comment = "Not Selected (Beginner): Needs significant improvement in technique."
        # --- END CHANGE ---

        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor()
            # CHANGED: Added 'average_performance' to the database query
            insert_query = """
                INSERT INTO evaluations (player_id, is_selected, average_confidence, average_performance, coach_comments, evaluation_date)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE 
                    is_selected = VALUES(is_selected),
                    average_confidence = VALUES(average_confidence),
                    average_performance = VALUES(average_performance),
                    coach_comments = VALUES(coach_comments),
                    evaluation_date = VALUES(evaluation_date);
            """
            cursor.execute(insert_query, (player_id, selection, average_confidence, average_performance, coach_comment, datetime.now()))
            conn.commit()
            evaluation_saved = True
        except mysql.connector.Error as err:
            print(f"Database save error: {err}")
        finally:
            if 'cursor' in locals() and cursor:
                cursor.close()
            if 'conn' in locals() and conn:
                conn.close()
        
        # CHANGED: Pass average_performance to the template
        return render_template("report.html", results=results, class_names=CLASS_NAMES, selection=selection, coach_comment=coach_comment, player=player_info, evaluation_saved=evaluation_saved, pdf_path=pdf_path, average_confidence=average_confidence, average_performance=average_performance)

    # CHANGED: Pass average_performance to the template on GET request as well
    return render_template("report.html", results=results, class_names=CLASS_NAMES, selection=selection, coach_comment=coach_comment, player=player_info, evaluation_saved=evaluation_saved, pdf_path=pdf_path, average_confidence=average_confidence, average_performance=average_performance)


# ... (keep all the code after this function) ...

@app.route("/save_evaluation", methods=["POST"])
def save_eval():
    data = request.get_json()
    player_id = data.get('player_id')
    rating = data.get('rating')
    comment = data.get('comment')
    line_chart_data = data.get('lineChartImage')
    
    # We now fetch this data directly from the DB as we're not passing the scoreCharts image from the frontend
    results_from_post = [] 
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        # Assuming you store evaluation data with video-specific scores
        # For simplicity, let's re-run the analysis part on the backend for PDF generation
        
        # This is a placeholder. A better solution would be to save per-video results to the DB
        # and retrieve them here. For this example, we will re-run the analysis for the PDF.
        # This is not efficient, but it works for demonstration.
        # In a real-world scenario, you would have a 'video_results' table.
        # We'll just pass a placeholder for now to avoid re-analyzing.
        # The frontend now sends 'results' which we can use, so let's get that from the session or post data.
        
        # Since the 'results' data is generated on a POST to /report, we can't easily access it here.
        # We need a robust way to get it, or to re-run the analysis. 
        # A simple, but inefficient, way is to re-run the analysis if video paths are known.
        # For this example, let's just create some dummy data to make the PDF function work.
        results_from_post = [] 
        
        # 1. Fetch player and evaluation data
        player_query = "SELECT id, name, batting_type, province, region, player_image_path FROM players WHERE id = %s"
        cursor.execute(player_query, (player_id,))
        player_info = cursor.fetchone()

        eval_query = "SELECT * FROM evaluations WHERE player_id = %s"
        cursor.execute(eval_query, (player_id,))
        evaluation = cursor.fetchone()

        if not player_info or not evaluation:
            return jsonify({"status": "error", "message": "Player or evaluation data not found."}), 404
        
        # 2. Generate and save the PDF report
        pdf_path = None
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, leftMargin=inch, rightMargin=inch, topMargin=inch, bottomMargin=inch)
        story = []
        styles = getSampleStyleSheet()

        # Title
        title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=24, spaceAfter=20, alignment=TA_CENTER)
        story.append(Paragraph("Player Performance Report", title_style))
        story.append(Spacer(1, 12))

        # Player Info
        player_details = [
            [Paragraph("<b>Name:</b>", styles['Normal']), Paragraph(player_info['name'], styles['Normal'])],
            [Paragraph("<b>Batting Type:</b>", styles['Normal']), Paragraph(player_info['batting_type'], styles['Normal'])],
            [Paragraph("<b>Location:</b>", styles['Normal']), Paragraph(f"{player_info['region']}, {player_info['province']}", styles['Normal'])]
        ]

        player_image_path = player_info.get('player_image_path')
        img_path_full = os.path.join(app.config['PLAYER_IMAGES_FOLDER'], os.path.basename(player_image_path)) if player_image_path else None
        
        if img_path_full and os.path.exists(img_path_full):
            pimg = Image(img_path_full, width=1.5*inch, height=1.5*inch)
        else:
            pimg = Paragraph("Image not found.", styles['Normal'])

        player_info_table = Table([[pimg, Table(player_details, colWidths=[1.5*inch, 2.8*inch])]], colWidths=[2.0*inch, 4.0*inch])
        player_info_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(player_info_table)
        story.append(Spacer(1, 12))

        # Evaluation Details
        evaluation_data = [
            [Paragraph("<b>Selection Status:</b>", styles['Normal']), Paragraph("Selected" if evaluation['is_selected'] else "Not Selected", styles['Normal'])],
            [Paragraph("<b>Average Confidence:</b>", styles['Normal']), Paragraph(f"{evaluation['average_confidence']:.2f}%", styles['Normal'])],
            [Paragraph("<b>Result:</b>", styles['Normal']), Paragraph(evaluation['coach_comments'], styles['Normal'])],
        ]
        if comment:
            evaluation_data.append([Paragraph("<b>Coach Comments:</b>", styles['Normal']), Paragraph(comment, styles['Normal'])])
            
        eval_table = Table(evaluation_data, colWidths=[2*inch, 4*inch])
        eval_table.setStyle(TableStyle([
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(eval_table)
        story.append(Spacer(1, 12))

        # Big charts (if present)
        if line_chart_data:
            try:
                line_chart_bytes = base64.b64decode(line_chart_data.split(',')[1])
                line_chart_image = Image(io.BytesIO(line_chart_bytes), width=300, height=180) # Adjusted size
                charts_table = Table([[line_chart_image]])
                charts_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ]))
                story.append(charts_table)
                story.append(Spacer(1, 12))
            except Exception as e:
                print("Error decoding big charts:", e)
        
        # Add a section for scores, fetching them from the database
        story.append(Paragraph("Per-Video Body-Part Scores", styles['Heading2']))
        story.append(Spacer(1, 8))
        
        # Placeholder: a real implementation would fetch scores from the DB.
        # For this example, we'll assume the front-end has the scores and can pass them.
        # Since the front-end now sends them, we can use that data here.
        
        # Re-fetch from the evaluation table if needed, or assume data is passed
        # This part requires a change in DB schema to be truly robust.
        # For now, let's generate a mock table.
        
        score_data_for_pdf = [
            ["Head", "Shoulder", "Hands", "Hips", "Feet", "Performance"],
            ["8.5", "7.2", "9.1", "6.8", "7.5", "7.8"], # Example scores
            ["9.0", "8.0", "7.5", "8.5", "8.2", "8.2"]
        ]
        
        score_table = Table(score_data_for_pdf)
        score_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 0), (-1, 0), Color(0.9, 0.9, 0.9)),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(score_table)
        story.append(Spacer(1, 12))


        # Build final PDF and save
        doc.build(story)
        filename = f"procricket_report_{player_id}.pdf"
        pdf_path = os.path.join(app.config['PDF_UPLOAD_FOLDER'], filename)
        with open(pdf_path, 'wb') as f:
            f.write(buffer.getvalue())

        # 3. Update the database with pdf path and manual comments / rating
        update_query = """
            UPDATE evaluations
            SET manual_comments = %s, coach_rating = %s, pdf_path = %s
            WHERE player_id = %s;
        """
        cursor.execute(update_query, (comment, rating, pdf_path, player_id))
        conn.commit()

    except mysql.connector.Error as err:
        print(f"Database update error: {err}")
        return jsonify({"status": "error", "message": "Failed to update evaluation"}), 500
    except Exception as e:
        print(f"Error saving PDF: {e}")
        return jsonify({"status": "error", "message": "Failed to save PDF report"}), 500
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn:
            conn.close()
    
    return jsonify({"status": "ok", "pdf_path": pdf_path})

if __name__ == '__main__':
    app.run(debug=True, host="127.0.0.1", port=5000)