from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Train the model on the COCO8 example dataset for 5 epochs
results = model.train(data="coco8.yaml", epochs=5, imgsz=640)

# Run inference with the YOLO11n model on the 'bus.jpg' image
results = model("bus.jpg")

# Save the image(s) with bounding boxes and labels
for r in results:
    r.save(filename="detected_bus.jpg")  # Save result to a specific file
