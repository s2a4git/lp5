# Mini Project: Colorizing Old B&W Images: color old black and white images to colorful images
import cv2
import numpy as np

# Load model
prototxt = "colorization_deploy_v2.prototxt"
model = "colorization_release_v2.caffemodel"
points = "pts_in_hull.npy"

net = cv2.dnn.readNetFromCaffe(prototxt, model)
pts = np.load(points)

# Add cluster centers to model
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")

pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Load grayscale image
image = cv2.imread("b&w.jpg")
scaled = image.astype("float32") / 255.0
lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

# Extract L channel
L = lab[:, :, 0]
L_resized = cv2.resize(L, (224, 224))
L_resized -= 50

# Predict color
net.setInput(cv2.dnn.blobFromImage(L_resized))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

# Resize to original
ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

# Combine with L channel
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized, 0, 1)

# Save output
cv2.imwrite("output.png", (colorized * 255).astype("uint8"))

print("Colorization done! Check output.png")
