import os

def count_images_per_class(test_dir):
    class_counts = {}

    for class_name in sorted(os.listdir(test_dir)):
        class_path = os.path.join(test_dir, class_name)
        if os.path.isdir(class_path):
            image_count = len([
                file for file in os.listdir(class_path)
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            ])
            class_counts[class_name] = image_count

    print("🧾 Image count per class in 'test/':\n")
    for class_name, count in class_counts.items():
        print(f"{class_name:25} → {count} images")

    print(f"\n✅ Total images in test set: {sum(class_counts.values())}")

# 👉 Replace this path with your actual test folder path
test_folder_path = "train/"

count_images_per_class(test_folder_path)
