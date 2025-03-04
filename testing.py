import training
import torch
import os
from PIL import Image
import io


if __name__ == "__main__":
    model = training.SkinLesionNN(8)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if os.path.exists("skin_lesion_cnn.pt"):
        model.load_state_dict(torch.load("skin_lesion_cnn.pt", weights_only=False))
        model = model.to(device)
        model.eval()
        print("Model loaded")
        print(model)
    else:
        print("Model not found")
        exit()

    image_path = "Data/" + input("Enter image filename: ")
    image_data = open(image_path, "rb").read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    transform = training.Trainer.get_transform()
    image = transform(image).to(device)

    age = int(input("Enter age: "))
    sex = int(input("Enter sex (0 for male, 1 for female): "))
    site_input = int(input("Enter site (0 for head/neck, 1 for upper extremity, 2 for posterior torso, 3 for anterior torso, 4 for lower extremity, 5 for palms/soles): "))
    site = [0, 0, 0, 0, 0, 0]
    site[site_input] = 1

    metadata = torch.tensor([age, sex, *site], dtype=torch.float).to(device)

    print(image)
    print(metadata)
    prediction = model.forward(image, metadata)
    print(prediction)

