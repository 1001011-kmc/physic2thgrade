import cv2
import mediapipe as mp

# 1. 파일 경로 (영어 경로)
img_path = r"C:\Users\User\Desktop\coding 2th grade\picture\he.png"

# 이미지 불러오기
img = cv2.imread(img_path)
if img is None:
    print("이미지 로드 실패. 파일 경로를 다시 확인해라.")
    exit()

# 2. MediaPipe Pose 설정
mp_pose = mp.solutions.pose
# model_complexity=2: 속도보다 정확도 최우선 (척추 데이터 분석용)
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# OpenCV BGR을 MediaPipe RGB로 변환 후 처리
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = pose.process(img_rgb)

# 3. 골격 뼈대 그리기 및 데이터 추출
if results.pose_landmarks:
    mp_draw.draw_landmarks(
        img, 
        results.pose_landmarks, 
        mp_pose.POSE_CONNECTIONS,
        mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
    )
    
    # 척추/골반 주요 랜드마크 좌표 (z축 포함)
    landmarks = results.pose_landmarks.landmark
    l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    
    print(f"왼쪽 어깨: x={l_shoulder.x:.3f}, y={l_shoulder.y:.3f}, z={l_shoulder.z:.3f}")
    print(f"왼쪽 골반: x={l_hip.x:.3f}, y={l_hip.y:.3f}, z={l_hip.z:.3f}")
    print("스켈레톤 추출 완료.")
else:
    print("사람 인식 실패.")

# 4. 결과 화면 출력
cv2.imshow("MediaPipe Skeleton", img)
cv2.waitKey(0)
cv2.destroyAllWindows()