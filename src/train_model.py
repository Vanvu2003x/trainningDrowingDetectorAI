from ultralytics import YOLO
import torch

def main():
    print("=== YOLOv8 Training Setup ===\n")

    # 1️⃣ Kiểm tra GPU
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        print(f"GPU(s) có sẵn: {n_gpu}")
        for i in range(n_gpu):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_mem = torch.cuda.get_device_properties(i).total_memory // 1024**3
            print(f"  GPU {i}: {gpu_name} - {gpu_mem}GB VRAM")
        
        default_device = 0
        device_input = input(f"Chọn GPU (0-{n_gpu-1}) hoặc -1 để dùng CPU [mặc định {default_device}]: ")
        device = int(device_input) if device_input.strip().isdigit() else default_device
    else:
        print("Không tìm thấy GPU, sẽ dùng CPU.")
        device = -1

    # 2️⃣ Gợi ý batch size
    if device == -1:
        suggested_batch = 4
    else:
        gpu_mem = torch.cuda.get_device_properties(device).total_memory // 1024**3
        if gpu_mem < 4:
            suggested_batch = 4
        elif gpu_mem < 8:
            suggested_batch = 8
        elif gpu_mem < 12:
            suggested_batch = 16
        else:
            suggested_batch = 32
    print(f"Gợi ý batch size: {suggested_batch}")

    batch_input = input(f"Nhập batch size (mặc định {suggested_batch}): ")
    batch = int(batch_input) if batch_input.strip().isdigit() else suggested_batch

    epochs_input = input("Nhập số epoch (mặc định 100): ")
    epochs = int(epochs_input) if epochs_input.strip().isdigit() else 100

    # 5️⃣ Xác nhận cấu hình
    print("\n=== Xác nhận cấu hình ===")
    print(f"Device   : {'CPU' if device==-1 else 'GPU '+str(device)}")
    print(f"Batch    : {batch}")
    print(f"Epochs   : {epochs}")
    confirm = input("Có muốn bắt đầu train với cấu hình này? [y/n]: ")
    if confirm.lower() != 'y':
        print("Đã hủy train.")
        return

    # 6️⃣ Load model & train
    model = YOLO("../model/yolov8m.pt")
    model.train(
        data="../data/data.yaml",
        epochs=epochs,
        imgsz=640,
        batch=batch,
        device=device,
        project="runs/train",
        name="yolo_drowning_detection",
        exist_ok=True
    )

    print("\nTraining đã bắt đầu!")

if __name__ == "__main__":
    main()
