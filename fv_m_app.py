### 1. Imports and class names setup ###
import gradio as gr
import os
import torch

# from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple, Dict

import torchvision
from torch import nn


def fv_m_app():

    global g_vit
    global g_vit_transforms
    global g_class_names
    global g_img

    # Setup class names
    g_class_names = ["pizza", "steak", "sushi"]

    ### 2. Model and transforms preparation ###

    # Create EffNetB2 model
    g_vit, g_vit_transforms = create_vit_model(
        num_classes=3,  # len(class_names) would also work
    )

    # Load saved weights
    g_vit.load_state_dict(
        torch.load(
            f="./demos/foodvision_mini/09_pretrained_vit_feature_extractor_pizza_steak_sushi_20_percent.pth",
            map_location=torch.device("cpu"),  # load to CPU
        )
    )

    ### 4. Gradio app ###

    # Create title, description and article strings
    title = "FoodVision Mini 🍕🥩🍣"
    description = "An EfficientNetB2 feature extractor computer vision model to classify images of food as pizza, steak or sushi."
    article = "Created at [09. PyTorch Model Deployment](https://www.learnpytorch.io/09_pytorch_model_deployment/)."

    # Create examples list from "examples/" directory
    example_list = [["demos/foodvision_mini/examples/" + example] for example in
                    os.listdir("demos/foodvision_mini/examples")]

    # Create the Gradio demo
    demo = gr.Interface(fn=predict,  # mapping function from input to output
                        inputs=gr.Image(type="pil"),  # what are the inputs?
                        outputs=[gr.Label(num_top_classes=3, label="Predictions"),  # what are the outputs?
                                 gr.Number(label="Prediction time (s)")],
                        # our fn has two outputs, therefore we have two outputs
                        # Create examples list from "examples/" directory
                        examples=example_list,
                        title=title,
                        description=description,
                        article=article)

    # Launch the demo!
    demo.launch()


def create_vit_model(num_classes: int = 3,
                     seed: int = 42):
    """Creates a ViT-B/16 feature extractor model and transforms.

    Args:
        num_classes (int, optional): number of target classes. Defaults to 3.
        seed (int, optional): random seed value for output layer. Defaults to 42.

    Returns:
        model (torch.nn.Module): ViT-B/16 feature extractor model.
        transforms (torchvision.transforms): ViT-B/16 image transforms.
    """
    # Create ViT_B_16 pretrained weights, transforms and model
    weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    transforms = weights.transforms()
    model = torchvision.models.vit_b_16(weights=weights)

    # Freeze all layers in model
    for param in model.parameters():
        param.requires_grad = False

    # Change classifier head to suit our needs (this will be trainable)
    torch.manual_seed(seed)
    model.heads = nn.Sequential(nn.Linear(in_features=768,  # keep this the same as original model
                                          out_features=num_classes))  # update to reflect target number of classes

    return model, transforms


### 3. Predict function ###

# Create predict function
def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken.
    """
    # Start the timer
    start_time = timer()

    # Transform the target image and add a batch dimension
    img = g_vit_transforms(img).unsqueeze(0)

    # Put model into evaluation mode and turn on inference mode
    g_vit.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_probs = torch.softmax(g_vit(img), dim=1)

    # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
    pred_labels_and_probs = {g_class_names[i]: float(pred_probs[0][i]) for i in range(len(g_class_names))}

    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)

    # Return the prediction dictionary and prediction time
    return pred_labels_and_probs, pred_time

