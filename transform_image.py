from torchvision import transforms as T
from PIL import Image

def transform(image_path):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])

    transform = T.Compose([
        T.Resize([600]), 
        normalize
    ])

    image = Image.open(image_path).convert("RGB")  # Ensure it's RGB format

    transformed_image = transform(image)

    return transformed_image
