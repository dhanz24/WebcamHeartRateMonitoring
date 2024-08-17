import numpy as np
import cv2
import sys
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.PlotModule import LivePlot
import cvzone
import time
from scipy.signal import butter, filtfilt

realWidth = 640
realHeight = 480
videoWidth = 160
videoHeight = 120
videoChannels = 3
videoFrameRate = 15

# Webcam Parameters
webcam = cv2.VideoCapture(0)
detector = FaceDetector()

webcam.set(3, realWidth)
webcam.set(4, realHeight)

# Color Magnification Parameters
levels = 3
alpha = 170
minFrequency = 1.0
maxFrequency = 2.0
bufferSize = 150
bufferIndex = 0

plotY = LivePlot(realWidth, realHeight, [60, 120], invert=True)

# Helper Methods
def buildGauss(frame, levels):
    pyramid = [frame]
    for level in range(levels):
        frame = cv2.pyrDown(frame)
        pyramid.append(frame)
    return pyramid

def reconstructFrame(pyramid, index, levels):
    filteredFrame = pyramid[index]
    for level in range(levels):
        filteredFrame = cv2.pyrUp(filteredFrame)
    filteredFrame = filteredFrame[:videoHeight, :videoWidth]
    return filteredFrame

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data, axis=0)
    return y

# Output Display Parameters
font = cv2.FONT_HERSHEY_SIMPLEX
loadingTextLocation = (30, 40)
bpmTextLocation = (videoWidth // 2, 40)
fpsTextLocation = (500, 600)

fontScale = 1
fontColor = (255, 255, 255)
lineType = 2
boxColor = (0, 255, 0)
boxWeight = 3

# Initialize Gaussian Pyramid
firstFrame = np.zeros((videoHeight, videoWidth, videoChannels))
firstGauss = buildGauss(firstFrame, levels + 1)[levels]
videoGauss = np.zeros((bufferSize, firstGauss.shape[0], firstGauss.shape[1], videoChannels))
fourierTransformAvg = np.zeros((bufferSize))

# Bandpass Filter for Specified Frequencies
frequencies = (1.0 * videoFrameRate) * np.arange(bufferSize) / (1.0 * bufferSize)
mask = (frequencies >= minFrequency) & (frequencies <= maxFrequency)

# Heart Rate Calculation Variables
bpmCalculationFrequency = 19
bpmBufferIndex = 0
bpmBufferSize = 19
bpmBuffer = np.zeros((bpmBufferSize))

i = 0
ptime = 0
ftime = 0
while True:
    ret, frame = webcam.read()
    if not ret:
        break

    frame, bboxs = detector.findFaces(frame, draw=False)
    frameDraw = frame.copy()
    ftime = time.time()
    fps = 1 / (ftime - ptime)
    ptime = ftime

    cv2.putText(frameDraw, f'FPS: {int(fps)}', (30, 440), 0, 1, [0, 0, 0], thickness=2, lineType=cv2.LINE_AA)

    if bboxs:
        x1, y1, w1, h1 = bboxs[0]['bbox']

        # Focus on the forehead area
        forehead_h = int(h1 * 0.2)  # 20% height of the bounding box
        forehead_y = y1
        forehead_x = x1 + int(w1 * 0.3)  # Focus on the center of the forehead
        forehead_w = int(w1 * 0.4)  # 40% width of the bounding box

        detectionFrame = frame[forehead_y:forehead_y + forehead_h, forehead_x:forehead_x + forehead_w]
        detectionFrame = cv2.resize(detectionFrame, (videoWidth, videoHeight))

        # Draw rectangle on the forehead area
        cv2.rectangle(frameDraw, (forehead_x, forehead_y), (forehead_x + forehead_w, forehead_y + forehead_h),
                      (255, 0, 255), 2)

        # Construct Gaussian Pyramid
        videoGauss[bufferIndex] = buildGauss(detectionFrame, levels + 1)[levels]
        fourierTransform = np.fft.fft(videoGauss, axis=0)

        # Bandpass Filter
        fourierTransform[mask == False] = 0

        # Butterworth Bandpass Filter for Smoothing
        filtered = butter_bandpass_filter(fourierTransform, minFrequency, maxFrequency, videoFrameRate, order=6)

        # Grab a Pulse
        if bufferIndex % bpmCalculationFrequency == 0:
            i += 1
            for buf in range(bufferSize):
                fourierTransformAvg[buf] = np.real(filtered[buf]).mean()
            hz = frequencies[np.argmax(fourierTransformAvg)]
            bpm = 60.0 * hz
            bpmBuffer[bpmBufferIndex] = bpm
            bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize

        # Amplify
        filtered = np.real(np.fft.ifft(filtered, axis=0))
        filtered = filtered * alpha

        # Reconstruct Resulting Frame
        filteredFrame = reconstructFrame(filtered, bufferIndex, levels)
        outputFrame = detectionFrame + filteredFrame
        outputFrame = cv2.convertScaleAbs(outputFrame)

        bufferIndex = (bufferIndex + 1) % bufferSize
        outputFrame_show = cv2.resize(outputFrame, (videoWidth // 2, videoHeight // 2))
        frameDraw[0:videoHeight // 2, (realWidth - videoWidth // 2):realWidth] = outputFrame_show

        bpm_value = bpmBuffer.mean()
        imgPlot = plotY.update(float(bpm_value))

        if i > bpmBufferSize:
            cvzone.putTextRect(frameDraw, f'BPM: {bpm_value:.2f}', bpmTextLocation, scale=2)
        else:
            cvzone.putTextRect(frameDraw, "Calculating BPM...", loadingTextLocation, scale=2)

    else:
        # Display message when no face is detected
        cvzone.putTextRect(frameDraw, "No face detected", (realWidth // 2 - 100, realHeight // 2 - 20), scale=2, thickness=2)
        imgPlot = plotY.update(0)  # Update plot with 0 when no face is detected

    # Stack the frame and the plot together
    imgStack = cvzone.stackImages([frameDraw, imgPlot], 2, 1)
    cv2.imshow("Heart Rate Monitor", imgStack)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
 