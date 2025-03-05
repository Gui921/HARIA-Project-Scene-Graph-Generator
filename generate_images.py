import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import os

from class_and_labels import OBJ_CLASSES

def show_image_boxes(image_tensor, boxes, file_name,box_labels = [],obj_classes = OBJ_CLASSES):
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    # Denormalize the image tensor
    image_tensor = image_tensor * std[:, None, None] + mean[:, None, None]

    # Get image dimensions (height and width)
    height, width = image_tensor.shape[1], image_tensor.shape[2]

    # Convert tensor to NumPy for visualization
    image_np = image_tensor.permute(1, 2, 0).clamp(0, 1).numpy()

    # Create a figure and axis
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(image_np)

    # Loop over each box and label
    for box, label in zip(boxes, box_labels):
        x_center, y_center, w, h = box
        x_center = x_center * width
        y_center = y_center * height
        w = w * width
        h = h * height

        # Calculate the top-left corner
        x1 = x_center - w / 2
        y1 = y_center - h / 2

        # Draw the bounding box
        rect = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # Add the label text above the box
        if box_labels != []:
            class_name = obj_classes[label]  # Get the class name from the index
            ax.text(
                x1, y1 - 5, class_name, 
                fontsize=8, color='yellow', 
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none')
            )

    # Remove axes for better visualization
    plt.axis('off')
    #plt.show()
    #return fig, ax
    create_folder()
    save_path = os.path.join('output', file_name)
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def show_image_boxes_without_labels(image_tensor, boxes, file_name):
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    # Denormalize the image tensor
    image_tensor = image_tensor * std[:, None, None] + mean[:, None, None]

    # Get image dimensions (height and width)
    height, width = image_tensor.shape[1], image_tensor.shape[2]

    # Convert tensor to NumPy for visualization
    image_np = image_tensor.permute(1, 2, 0).clamp(0, 1).numpy()

    # Create a figure and axis
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(image_np)

    # Loop over each box and label
    for box in boxes:
        x_center, y_center, w, h = box
        x_center = x_center * width
        y_center = y_center * height
        w = w * width
        h = h * height

        # Calculate the top-left corner
        x1 = x_center - w / 2
        y1 = y_center - h / 2

        # Draw the bounding box
        rect = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)


    # Remove axes for better visualization
    plt.axis('off')
    #plt.show()
    #return fig, ax
    create_folder()
    save_path = os.path.join('output', file_name)
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def create_folder():

    if not os.path.exists("output"): 
        os.makedirs("output")





