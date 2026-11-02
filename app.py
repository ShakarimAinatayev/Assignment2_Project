import time
import gradio as gr
import matplotlib.cm as cm
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = "efficientnetv2_symbol_classifier.keras"

CLASS_NAMES = [
    "equals",
    "minus",
    "plus"
]


model = tf.keras.models.load_model(MODEL_PATH)



base_model = None

for layer in model.layers:
    if "efficientnet" in layer.name.lower():
        base_model = layer
        break

if base_model is None:
    raise RuntimeError("EfficientNetV2 backbone was not found.")


last_conv_layer = base_model.get_layer("top_conv")
base_model_index = model.layers.index(base_model)


def make_gradcam(img_array):
    feature_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[
            last_conv_layer.output,
            base_model.output
        ]
    )

    with tf.GradientTape() as tape:
        x = img_array


        for layer in model.layers[:base_model_index]:
            x = layer(x, training=False)


        conv_outputs, x = feature_model(
            x,
            training=False
        )


        for layer in model.layers[base_model_index + 1:]:
            x = layer(x, training=False)

        predictions = x

        predicted_index = tf.argmax(predictions[0])

        class_score = predictions[:, predicted_index]

    gradients = tape.gradient(
        class_score,
        conv_outputs
    )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)

    heatmap = heatmap / (
        tf.reduce_max(heatmap) + 1e-8
    )

    return heatmap.numpy()


def predict_symbol(image):
    if image is None:
        return "Please upload an image.", None, ""

    image = Image.fromarray(image).convert("RGB")
    image = image.resize((224, 224))

    original_image = np.array(image)

    img_array = original_image.astype("float32")
    batch = np.expand_dims(img_array, axis=0)


    start_time = time.perf_counter()

    predictions = model.predict(
        batch,
        verbose=0
    )

    latency_ms = (
        time.perf_counter() - start_time
    ) * 1000

    predicted_index = int(
        np.argmax(predictions[0])
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        predictions[0][predicted_index]
    )


    heatmap = make_gradcam(batch)

    heatmap = tf.image.resize(
        heatmap[..., np.newaxis],
        (224, 224)
    ).numpy().squeeze()

    jet = cm.get_cmap("jet")

    colored_heatmap = (
        jet(heatmap)[:, :, :3] * 255
    ).astype("uint8")

    overlay = (
        0.55 * original_image
        + 0.45 * colored_heatmap
    ).astype("uint8")

    prediction_text = (
        f"Prediction: {predicted_class}\n"
        f"Confidence: {confidence * 100:.2f}%"
    )

    latency_text = (
        f"Inference latency: {latency_ms:.2f} ms"
    )

    return prediction_text, overlay, latency_text


demo = gr.Interface(
    fn=predict_symbol,

    inputs=gr.Image(
        type="numpy",
        sources=["upload", "webcam"],
        label="Upload or capture a handwritten symbol"
    ),

    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Image(label="Grad-CAM Explanation"),
        gr.Textbox(label="Latency")
    ],

    title="Handwritten Symbol Classifier",

    description=(
        "Upload or capture a handwritten +, - or = symbol. "
        "The model predicts the class and shows a Grad-CAM explanation."
    )
)


if __name__ == "__main__":
    demo.launch()