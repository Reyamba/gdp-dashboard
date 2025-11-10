import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io

# --- Configuration ---
IMG_SIZE = 224
CLASS_NAMES = ['Healthy Cocoa', 'Black Pod Disease', 'Frosty Pod Rot', 'Witches Broom']
LAST_CONV_LAYER_NAME = 'conv_last'

# --- 1. Model Definition (Architecture Only) ---
@st.cache_resource
def build_and_initialize_model():
    """Defines a simple CNN architecture for cocoa detection."""
    input_tensor = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_tensor)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Last convolutional layer for Grad-CAM
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name=LAST_CONV_LAYER_NAME)(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    output_tensor = Dense(len(CLASS_NAMES), activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=output_tensor)
    
    # NOTE: In a real app, you would load pre-trained weights here:
    # try:
    #     model.load_weights('cocoa_detection_weights.h5')
    # except:
    #     st.warning("Weights not found. Using randomly initialized model for demonstration.")
    
    st.success("CNN Model loaded successfully (using random weights).")
    return model

# --- 2. Grad-CAM Implementation ---

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Computes and returns the Grad-CAM heatmap.
    Based on the official Keras Grad-CAM example.
    """
    # Create a model that maps the input image to the activations of the last conv layer
    # and the final output layer.
    grad_model = Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Use tf.GradientTape to compute gradients
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array[np.newaxis, ...])
        
        # If no specific index is provided, use the index of the highest prediction
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        
        # Get the loss for the predicted class
        class_channel = preds[:, pred_index]

    # Compute the gradient of the predicted class output with respect to the last conv layer output
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # Compute the spatial average of the gradients (Global Average Pooling)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Multiply the activation map by the weights (pooled gradients)
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    
    return heatmap.numpy(), pred_index

def display_gradcam(img, heatmap, alpha=0.4):
    """Overlays the heatmap on the original image."""
    
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use colormap jet to generate the colors for the heatmap
    jet = plt.cm.get_cmap("jet")
    
    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on the original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)
    
    return superimposed_img


# --- 3. Streamlit App Functions ---

def preprocess_image(image_file):
    """Load, resize, and normalize the image."""
    img = Image.open(image_file).convert('RGB')
    
    # Keep original image for overlay
    img_array_original = np.array(img.resize((IMG_SIZE, IMG_SIZE)))
    
    # Preprocess for model
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = keras.utils.img_to_array(img)
    img_array = img_array / 255.0  # Normalize to 0-1
    
    return img_array, img_array_original

# --- Streamlit UI Layout ---

st.set_page_config(
    page_title="Cocoa Disease Detector (CNN + Grad-CAM)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the model once
model = build_and_initialize_model()

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f7f3e8; /* Light cocoa/cream background */
    }
    .header-title {
        color: #4b2c15; /* Dark chocolate text */
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 2.5em;
    }
    .st-emotion-cache-1pxn4lb { /* Adjust main content padding */
        padding-top: 2rem;
    }
    .stButton>button {
        background-color: #795548; /* Brown button */
        color: white;
        border-radius: 12px;
        border: none;
        padding: 10px 20px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #5d4037;
        transform: translateY(-2px);
        box-shadow: 4px 4px 8px rgba(0,0,0,0.3);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="header-title">🍫 Cocoa Disease AI Detector & Explainer</div>', unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Upload Image")
    st.markdown("Upload a photo of a cocoa pod or leaf to detect potential diseases.")
    uploaded_file = st.file_uploader("Choose an Image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Cocoa Image', use_column_width=True)
        
        # Process and Predict
        img_array_norm, img_array_orig = preprocess_image(uploaded_file)
        
        if st.button('Analyze Cocoa Image'):
            # Run Prediction
            with st.spinner('Analyzing image and calculating Grad-CAM...'):
                predictions = model.predict(img_array_norm[np.newaxis, ...])[0]
                pred_index = np.argmax(predictions)
                
                # Run Grad-CAM
                heatmap, final_pred_index = make_gradcam_heatmap(
                    img_array_norm, 
                    model, 
                    LAST_CONV_LAYER_NAME, 
                    pred_index=pred_index
                )
                
                gradcam_img = display_gradcam(img_array_orig, heatmap)
            
            # Save results to session state
            st.session_state['pred_class'] = CLASS_NAMES[pred_index]
            st.session_state['confidence'] = predictions[pred_index]
            st.session_state['all_predictions'] = dict(zip(CLASS_NAMES, predictions))
            st.session_state['gradcam_img'] = gradcam_img
            
with col2:
    st.header("Detection Results")
    st.markdown("The model's classification and its area of focus (Grad-CAM) will appear here.")
    
    if 'pred_class' in st.session_state:
        st.subheader("Classification:")
        
        # Display main result
        pred_class = st.session_state['pred_class']
        confidence = st.session_state['confidence']
        st.markdown(f"**Result:** <span style='font-size: 1.5em; color: #d32f2f; font-weight: bold;'>{pred_class}</span>", unsafe_allow_html=True)
        st.markdown(f"**Confidence:** `{confidence * 100:.2f}%`")
        
        st.subheader("Model Explanation (Grad-CAM):")
        
        # Display Grad-CAM image
        st.image(
            st.session_state['gradcam_img'], 
            caption=f'Grad-CAM Heatmap: Highlights the areas the model focused on for the "{pred_class}" prediction.', 
            use_column_width=True
        )

        st.info("The colored overlay (heatmap) shows the regions of the image that contributed most significantly to the model's prediction. Red/Yellow areas indicate high influence.")
        
        # Display all probabilities
        st.subheader("All Probabilities")
        all_preds = st.session_state['all_predictions']
        
        # Convert dictionary to a list of (class, prob) tuples and sort by probability
        data = sorted(
            [(k, v) for k, v in all_preds.items()], 
            key=lambda item: item[1], 
            reverse=True
        )
        
        # Create a simple table
        st.dataframe(
            data=data,
            column_config={
                0: st.column_config.TextColumn("Disease/State", width="large"),
                1: st.column_config.NumberColumn("Probability", format="%.4f", width="small")
            },
            hide_index=True,
            use_container_width=True
        )

# --- Instructions for the user ---
st.markdown("---")
st.markdown("### How to Run This App")
st.code("streamlit run streamlit_cocoa_detector.py")
st.markdown("1. Save the code above as `streamlit_cocoa_detector.py`.")
st.markdown("2. Ensure you have the required libraries installed: `pip install streamlit tensorflow numpy matplotlib Pillow`.")
st.markdown("3. Run the command above in your terminal.")    },
    "Witches_Broom": {
        "description": "Witches' Broom disease, caused by the fungus Moniliophthora perniciosa, leads to abnormal, dense growths of shoots or 'brooms' and swollen stems. It can also affect pods, causing premature ripening and malformation.",
        "causes": "Fungal infection (Moniliophthora perniciosa).",
        "treatment": "Pruning infected parts, resistant varieties, fungicides. Strict phytosanitary measures are crucial.",
        "severity": "High - can significantly reduce yield and lead to tree death if untreated."
    },
    "Black_Pod_Rot": {
        "description": "Black Pod Rot, caused by various Phytophthora species, is characterized by rapidly spreading black or brown lesions on cacao pods, eventually covering the entire pod. It also affects flowers, cushions, and leaves.",
        "causes": "Fungal-like oomycete infection (Phytophthora spp.), favored by high humidity and rainfall.",
        "treatment": "Sanitation (removing infected pods), copper-based fungicides, good drainage, wider spacing of trees. Breeding for resistance is ongoing.",
        "severity": "High - major cause of yield loss globally."
    },
    "Frosty_Pod_Rot": {
        "description": "Frosty Pod Rot, caused by the fungus Moniliophthora roreri, initially appears as small, water-soaked spots on pods, which then develop a white, powdery fungal growth (like frost). The internal beans turn black and rot, making the pods commercially worthless.",
        "causes": "Fungal infection (Moniliophthora roreri), spread by wind and rain.",
        "treatment": "Strict sanitation (removing mummified pods), shade management, pruning, and developing resistant varieties. Fungicides are generally less effective once symptoms appear.",
        "severity": "High - causes severe economic losses in affected regions."
    }
}
"""
DISEASE_DESCRIPTIONS = json.loads(DISEASE_DESCRIPTIONS_JSON)

NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 4 
EPOCHS = 15 


@st.cache_data
def load_images_from_web_simulated():
    """
    Simulates loading a synthetic, balanced dataset for training purposes.
    """
    
    NUM_SAMPLES_PER_CLASS = 12 
    
    X = []
    labels = []
    
    label_to_index = {name: i for i, name in enumerate(CLASS_NAMES)}
    
    for label_name in CLASS_NAMES:
        for _ in range(NUM_SAMPLES_PER_CLASS):
            # Generate a dummy image 
            dummy_img = np.random.rand(IMAGE_SIZE[0], IMAGE_SIZE[1], 3).astype(np.float32) * 255.0
            X.append(dummy_img)
            labels.append(label_name)

    X = np.array(X, dtype=np.float32) / 255.0 # Normalize dummy data
    label_indices = np.array([label_to_index[label] for label in labels])
    y = tf.keras.utils.to_categorical(label_indices, num_classes=NUM_CLASSES)
    
    return X, y

# --- 2. Model Definition and Training ---

@st.cache_resource
def build_and_train_model():
    """Builds, compiles, and trains the CNN model."""
    st.info("Simulating data loading and processing...")
    
    X, y = load_images_from_web_simulated()

    # Split the simulated dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=np.argmax(y, axis=1)
    )

    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', name='last_conv_layer'), # Target layer for Grad-CAM
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    st.info("Training CNN Model... (Using simulated data)")
    
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        verbose=0
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    try:
        report = classification_report(y_true_classes, y_pred_classes, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    except ValueError:
        report = {'accuracy': 0.0, 'macro avg': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0}}
    
    conf_mat = confusion_matrix(y_true_classes, y_pred_classes)

    return model, history, acc, report, conf_mat, X_test, y_test

# --- 3. Advanced Grad-CAM Implementation ---

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Computes and returns the Grad-CAM heatmap."""
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array[np.newaxis, ...]) 
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize heatmap
    if tf.reduce_max(heatmap) > 1e-8:
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    else:
        heatmap = tf.zeros_like(heatmap)
        
    heatmap = heatmap.numpy()

    return heatmap, preds[0].numpy()

def display_gradcam(img, heatmap, alpha=0.5):
    """Overlays the heatmap on the original image."""
    
    # Check for OpenCV (cv2) global availability
    if 'cv2' not in globals():
        return None, "OpenCV (cv2) is not available to generate the Grad-CAM visualization."

    heatmap = np.uint8(255 * heatmap)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
    resized_heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    colormap = cv2.COLORMAP_JET
    heatmap_jet = cv2.applyColorMap(resized_heatmap, colormap)

    superimposed_img_bgr = cv2.addWeighted(img_bgr, 1.0 - alpha, heatmap_jet, alpha, 0)
    superimposed_img_rgb = cv2.cvtColor(superimposed_img_bgr, cv2.COLOR_BGR2RGB)

    # Encode image to bytes for Streamlit display
    is_success, buffer = cv2.imencode(".png", cv2.cvtColor(superimposed_img_rgb, cv2.COLOR_BGR2RGB))
    if is_success:
        return buffer.tobytes(), None
    return None, "Failed to encode Grad-CAM image."

def process_uploaded_file(uploaded_file, model):
    """Handles uploaded file, runs prediction, and generates Grad-CAM."""
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
    if 'cv2' not in globals():
        st.error("Cannot proceed: OpenCV (cv2) is required for image processing.")
        return "N/A", 0.0, None

    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # Ensure image is not None (decoding failed)
    if img_bgr is None:
        st.error("Uploaded file could not be decoded as an image. Please check the file format.")
        return "N/A", 0.0, None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img_model_input = cv2.resize(img_rgb, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    img_array = img_model_input.astype('float32') / 255.0

    LAST_CONV_LAYER_NAME = 'last_conv_layer'
    heatmap, preds = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER_NAME)

    predicted_class_index = np.argmax(preds)
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = preds[predicted_class_index] * 100

    gradcam_img_bytes, error = display_gradcam(img_rgb, heatmap, alpha=0.5)
    
    if error:
        st.error(error)
        return predicted_class, confidence, None

    return predicted_class, confidence, gradcam_img_bytes

# --- 4. Streamlit Application (Dashboard) ---

def main():
    st.set_page_config(layout="wide", page_title="Advanced Cacao Disease Detector")

    st.title("🌿 Cacao Disease Detector & Explainable AI Dashboard")
    st.markdown("Upload an image of your cacao plant or pod to detect common diseases (or confirm health) and see **where** the model is looking with Grad-CAM.")
    st.info("⚠️ This model is trained on **simulated data** for demonstration. For real use, you must supply your own large, labeled image dataset.")
    
    # --- Sidebar for Model Metrics ---
    with st.sidebar:
        st.title("Model Training & Metrics")
        st.info(f"Model trained on **{NUM_CLASSES}** classes: {', '.join(CLASS_NAMES)}")

        with st.spinner("Building and training the model..."):
            # Load and train the model (cached)
            model, history, acc, report, conf_mat, X_test, y_test = build_and_train_model()
            st.success("Model trained successfully!")

        st.header("Model Validation Results (Simulated)")
        st.markdown(f"**Test Set Accuracy:** **{acc:.2f}**")
        
        if history.history:
            st.markdown(f"**Test Loss:** **{history.history['val_loss'][-1]:.4f}**")

        with st.expander("Detailed Classification Report"):
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)

        if SEABORN_AVAILABLE:
            with st.expander("Confusion Matrix"):
                fig, ax = plt.subplots(figsize=(6, 6))
                
                y_pred_classes_cmat = np.argmax(model.predict(X_test, verbose=0), axis=1)
                y_true_classes_cmat = np.argmax(y_test, axis=1)

                cmat_disp = confusion_matrix(y_true_classes_cmat, y_pred_classes_cmat)
                
                sns.heatmap(cmat_disp, annot=True, fmt='d', cmap='Blues', cbar=False,
                            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                ax.set_title('Confusion Matrix (Simulated)')
                st.pyplot(fig)
        else:
            st.warning("Seaborn is required to display the Confusion Matrix plot.")

    # --- Main App for Prediction ---
    st.header("Upload Cacao Image for Analysis")

    # Image Upload Widget (The user dashboard component)
    uploaded_file = st.file_uploader(
        "Upload an image of a cacao leaf or pod (JPEG/PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        st.subheader("Analysis Results")
        
        # Create responsive columns for the results display
        col_img, col_gradcam, col_description = st.columns([1, 1, 1.5]) 

        # Process and Predict
        predicted_class, confidence, gradcam_img_bytes = process_uploaded_file(uploaded_file, model)
        
        # COLUMN 1: Original Image and Prediction
        with col_img:
            st.markdown("<p style='text-align: center; font-size: 1.25rem; font-weight: bold;'>Your Uploaded Image</p>", unsafe_allow_html=True)
            st.image(uploaded_file, use_column_width=True)
            st.markdown(f"**Predicted Status:** <span style='color: #1E88E5; font-size: 1.5rem;'>**{predicted_class}**</span>", unsafe_allow_html=True)
            st.markdown(f"**Confidence:** <span style='color: #4CAF50; font-size: 1.5rem;'>**{confidence:.2f}%**</span>", unsafe_allow_html=True)

        # COLUMN 2: Grad-CAM Visualization
        with col_gradcam:
            st.markdown("<p style='text-align: center; font-size: 1.25rem; font-weight: bold;'>Model Focus (Grad-CAM)</p>", unsafe_allow_html=True)
            if gradcam_img_bytes:
                base64_img = base64.b64encode(gradcam_img_bytes).decode('utf-8')
                st.image(f"data:image/png;base64,{base64_img}", use_column_width=True)
                st.markdown('''
                <small>The **red/yellow areas** indicate where the model focused its attention to make the prediction. This acts like an **arrow pointing to the key visual evidence** for the detected disease or health.</small>
                ''', unsafe_allow_html=True)
            else:
                st.error("Could not generate Grad-CAM visualization. Ensure OpenCV is correctly installed.")
        
        # COLUMN 3: Disease Description
        with col_description:
            st.markdown("<p style='text-align: center; font-size: 1.25rem; font-weight: bold;'>Detailed Information</p>", unsafe_allow_html=True)
            if predicted_class in DISEASE_DESCRIPTIONS:
                disease_info = DISEASE_DESCRIPTIONS[predicted_class]
                st.markdown(f"### {predicted_class.replace('_', ' ')}")
                st.markdown(f"**Description:** {disease_info.get('description', 'No detailed description available.')}")
                if disease_info.get('causes'):
                    st.markdown(f"**Causes:** {disease_info['causes']}")
                if disease_info.get('treatment'):
                    st.markdown(f"**Treatment/Management:** {disease_info['treatment']}")
                if disease_info.get('severity'):
                    st.markdown(f"**Severity:** {disease_info['severity']}")
            else:
                st.info("No specific description available for this prediction.")

    else:
        st.info("Please upload an image of a cacao plant or pod to get started!")

# Call main function without the conditional block
main()    },
    "Witches_Broom": {
        "description": "Witches' Broom disease, caused by the fungus Moniliophthora perniciosa, leads to abnormal, dense growths of shoots or 'brooms' and swollen stems. It can also affect pods, causing premature ripening and malformation.",
        "causes": "Fungal infection (Moniliophthora perniciosa).",
        "treatment": "Pruning infected parts, resistant varieties, fungicides. Strict phytosanitary measures are crucial.",
        "severity": "High - can significantly reduce yield and lead to tree death if untreated."
    },
    "Black_Pod_Rot": {
        "description": "Black Pod Rot, caused by various Phytophthora species, is characterized by rapidly spreading black or brown lesions on cacao pods, eventually covering the entire pod. It also affects flowers, cushions, and leaves.",
        "causes": "Fungal-like oomycete infection (Phytophthora spp.), favored by high humidity and rainfall.",
        "treatment": "Sanitation (removing infected pods), copper-based fungicides, good drainage, wider spacing of trees. Breeding for resistance is ongoing.",
        "severity": "High - major cause of yield loss globally."
    },
    "Frosty_Pod_Rot": {
        "description": "Frosty Pod Rot, caused by the fungus Moniliophthora roreri, initially appears as small, water-soaked spots on pods, which then develop a white, powdery fungal growth (like frost). The internal beans turn black and rot, making the pods commercially worthless.",
        "causes": "Fungal infection (Moniliophthora roreri), spread by wind and rain.",
        "treatment": "Strict sanitation (removing mummified pods), shade management, pruning, and developing resistant varieties. Fungicides are generally less effective once symptoms appear.",
        "severity": "High - causes severe economic losses in affected regions."
    }
}
"""
DISEASE_DESCRIPTIONS = json.loads(DISEASE_DESCRIPTIONS_JSON)

NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 4 
EPOCHS = 15 


@st.cache_data
def load_images_from_web_simulated():
    """
    Simulates loading a synthetic, balanced dataset for training purposes.
    """
    
    NUM_SAMPLES_PER_CLASS = 12 
    
    X = []
    labels = []
    
    label_to_index = {name: i for i, name in enumerate(CLASS_NAMES)}
    
    for label_name in CLASS_NAMES:
        for _ in range(NUM_SAMPLES_PER_CLASS):
            # Generate a dummy image 
            dummy_img = np.random.rand(IMAGE_SIZE[0], IMAGE_SIZE[1], 3).astype(np.float32) * 255.0
            X.append(dummy_img)
            labels.append(label_name)

    X = np.array(X, dtype=np.float32) / 255.0 # Normalize dummy data
    label_indices = np.array([label_to_index[label] for label in labels])
    y = tf.keras.utils.to_categorical(label_indices, num_classes=NUM_CLASSES)
    
    return X, y

# --- 2. Model Definition and Training ---

@st.cache_resource
def build_and_train_model():
    """Builds, compiles, and trains the CNN model."""
    st.info("Simulating data loading and processing...")
    
    X, y = load_images_from_web_simulated()

    # Split the simulated dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=np.argmax(y, axis=1)
    )

    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', name='last_conv_layer'), # Target layer for Grad-CAM
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    st.info("Training CNN Model... (Using simulated data)")
    
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        verbose=0
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    try:
        report = classification_report(y_true_classes, y_pred_classes, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    except ValueError:
        report = {'accuracy': 0.0, 'macro avg': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0}}
    
    conf_mat = confusion_matrix(y_true_classes, y_pred_classes)

    return model, history, acc, report, conf_mat, X_test, y_test

# --- 3. Advanced Grad-CAM Implementation ---

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Computes and returns the Grad-CAM heatmap."""
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array[np.newaxis, ...]) 
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize heatmap
    if tf.reduce_max(heatmap) > 1e-8:
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    else:
        heatmap = tf.zeros_like(heatmap)
        
    heatmap = heatmap.numpy()

    return heatmap, preds[0].numpy()

def display_gradcam(img, heatmap, alpha=0.5):
    """Overlays the heatmap on the original image."""
    
    # Check for OpenCV (cv2) global availability
    if 'cv2' not in globals():
        return None, "OpenCV (cv2) is not available to generate the Grad-CAM visualization."

    heatmap = np.uint8(255 * heatmap)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
    resized_heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    colormap = cv2.COLORMAP_JET
    heatmap_jet = cv2.applyColorMap(resized_heatmap, colormap)

    superimposed_img_bgr = cv2.addWeighted(img_bgr, 1.0 - alpha, heatmap_jet, alpha, 0)
    superimposed_img_rgb = cv2.cvtColor(superimposed_img_bgr, cv2.COLOR_BGR2RGB)

    # Encode image to bytes for Streamlit display
    is_success, buffer = cv2.imencode(".png", cv2.cvtColor(superimposed_img_rgb, cv2.COLOR_BGR2RGB))
    if is_success:
        return buffer.tobytes(), None
    return None, "Failed to encode Grad-CAM image."

def process_uploaded_file(uploaded_file, model):
    """Handles uploaded file, runs prediction, and generates Grad-CAM."""
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
    if 'cv2' not in globals():
        st.error("Cannot proceed: OpenCV (cv2) is required for image processing.")
        return "N/A", 0.0, None

    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # Ensure image is not None (decoding failed)
    if img_bgr is None:
        st.error("Uploaded file could not be decoded as an image. Please check the file format.")
        return "N/A", 0.0, None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img_model_input = cv2.resize(img_rgb, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    img_array = img_model_input.astype('float32') / 255.0

    LAST_CONV_LAYER_NAME = 'last_conv_layer'
    heatmap, preds = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER_NAME)

    predicted_class_index = np.argmax(preds)
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = preds[predicted_class_index] * 100

    gradcam_img_bytes, error = display_gradcam(img_rgb, heatmap, alpha=0.5)
    
    if error:
        st.error(error)
        return predicted_class, confidence, None

    return predicted_class, confidence, gradcam_img_bytes

# --- 4. Streamlit Application (Dashboard) ---

def main():
    st.set_page_config(layout="wide", page_title="Advanced Cacao Disease Detector")

    st.title("🌿 Cacao Disease Detector & Explainable AI Dashboard")
    st.markdown("Upload an image of your cacao plant or pod to detect common diseases (or confirm health) and see **where** the model is looking with Grad-CAM.")
    st.info("⚠️ This model is trained on **simulated data** for demonstration. For real use, you must supply your own large, labeled image dataset.")
    
    # --- Sidebar for Model Metrics ---
    with st.sidebar:
        st.title("Model Training & Metrics")
        st.info(f"Model trained on **{NUM_CLASSES}** classes: {', '.join(CLASS_NAMES)}")

        with st.spinner("Building and training the model..."):
            # Load and train the model (cached)
            model, history, acc, report, conf_mat, X_test, y_test = build_and_train_model()
            st.success("Model trained successfully!")

        st.header("Model Validation Results (Simulated)")
        st.markdown(f"**Test Set Accuracy:** **{acc:.2f}**")
        
        if history.history:
            st.markdown(f"**Test Loss:** **{history.history['val_loss'][-1]:.4f}**")

        with st.expander("Detailed Classification Report"):
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)

        if SEABORN_AVAILABLE:
            with st.expander("Confusion Matrix"):
                fig, ax = plt.subplots(figsize=(6, 6))
                
                y_pred_classes_cmat = np.argmax(model.predict(X_test, verbose=0), axis=1)
                y_true_classes_cmat = np.argmax(y_test, axis=1)

                cmat_disp = confusion_matrix(y_true_classes_cmat, y_pred_classes_cmat)
                
                sns.heatmap(cmat_disp, annot=True, fmt='d', cmap='Blues', cbar=False,
                            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                ax.set_title('Confusion Matrix (Simulated)')
                st.pyplot(fig)
        else:
            st.warning("Seaborn is required to display the Confusion Matrix plot.")

    # --- Main App for Prediction ---
    st.header("Upload Cacao Image for Analysis")

    # Image Upload Widget (The user dashboard component)
    uploaded_file = st.file_uploader(
        "Upload an image of a cacao leaf or pod (JPEG/PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        st.subheader("Analysis Results")
        
        # Create responsive columns for the results display
        col_img, col_gradcam, col_description = st.columns([1, 1, 1.5]) 

        # Process and Predict
        predicted_class, confidence, gradcam_img_bytes = process_uploaded_file(uploaded_file, model)
        
        # COLUMN 1: Original Image and Prediction
        with col_img:
            st.markdown("<p style='text-align: center; font-size: 1.25rem; font-weight: bold;'>Your Uploaded Image</p>", unsafe_allow_html=True)
            st.image(uploaded_file, use_column_width=True)
            st.markdown(f"**Predicted Status:** <span style='color: #1E88E5; font-size: 1.5rem;'>**{predicted_class}**</span>", unsafe_allow_html=True)
            st.markdown(f"**Confidence:** <span style='color: #4CAF50; font-size: 1.5rem;'>**{confidence:.2f}%**</span>", unsafe_allow_html=True)

        # COLUMN 2: Grad-CAM Visualization
        with col_gradcam:
            st.markdown("<p style='text-align: center; font-size: 1.25rem; font-weight: bold;'>Model Focus (Grad-CAM)</p>", unsafe_allow_html=True)
            if gradcam_img_bytes:
                base64_img = base64.b64encode(gradcam_img_bytes).decode('utf-8')
                st.image(f"data:image/png;base64,{base64_img}", use_column_width=True)
                st.markdown('''
                <small>The **red/yellow areas** indicate where the model focused its attention to make the prediction. This acts like an **arrow pointing to the key visual evidence** for the detected disease or health.</small>
                ''', unsafe_allow_html=True)
            else:
                st.error("Could not generate Grad-CAM visualization. Ensure OpenCV is correctly installed.")
        
        # COLUMN 3: Disease Description
        with col_description:
            st.markdown("<p style='text-align: center; font-size: 1.25rem; font-weight: bold;'>Detailed Information</p>", unsafe_allow_html=True)
            if predicted_class in DISEASE_DESCRIPTIONS:
                disease_info = DISEASE_DESCRIPTIONS[predicted_class]
                st.markdown(f"### {predicted_class.replace('_', ' ')}")
                st.markdown(f"**Description:** {disease_info.get('description', 'No detailed description available.')}")
                if disease_info.get('causes'):
                    st.markdown(f"**Causes:** {disease_info['causes']}")
                if disease_info.get('treatment'):
                    st.markdown(f"**Treatment/Management:** {disease_info['treatment']}")
                if disease_info.get('severity'):
                    st.markdown(f"**Severity:** {disease_info['severity']}")
            else:
                st.info("No specific description available for this prediction.")

    else:
        st.info("Please upload an image of a cacao plant or pod to get started!")

if __name__ == '__main__':
    main()    },
    "Witches_Broom": {
        "description": "Witches' Broom disease, caused by the fungus Moniliophthora perniciosa, leads to abnormal, dense growths of shoots or 'brooms' and swollen stems. It can also affect pods, causing premature ripening and malformation.",
        "causes": "Fungal infection (Moniliophthora perniciosa).",
        "treatment": "Pruning infected parts, resistant varieties, fungicides. Strict phytosanitary measures are crucial.",
        "severity": "High - can significantly reduce yield and lead to tree death if untreated."
    },
    "Black_Pod_Rot": {
        "description": "Black Pod Rot, caused by various Phytophthora species, is characterized by rapidly spreading black or brown lesions on cacao pods, eventually covering the entire pod. It also affects flowers, cushions, and leaves.",
        "causes": "Fungal-like oomycete infection (Phytophthora spp.), favored by high humidity and rainfall.",
        "treatment": "Sanitation (removing infected pods), copper-based fungicides, good drainage, wider spacing of trees. Breeding for resistance is ongoing.",
        "severity": "High - major cause of yield loss globally."
    },
    "Frosty_Pod_Rot": {
        "description": "Frosty Pod Rot, caused by the fungus Moniliophthora roreri, initially appears as small, water-soaked spots on pods, which then develop a white, powdery fungal growth (like frost). The internal beans turn black and rot, making the pods commercially worthless.",
        "causes": "Fungal infection (Moniliophthora roreri), spread by wind and rain.",
        "treatment": "Strict sanitation (removing mummified pods), shade management, pruning, and developing resistant varieties. Fungicides are generally less effective once symptoms appear.",
        "severity": "High - causes severe economic losses in affected regions."
    }
}
"""
DISEASE_DESCRIPTIONS = json.loads(DISEASE_DESCRIPTIONS_JSON)

NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 4 
EPOCHS = 15 


@st.cache_data
def load_images_from_web_simulated():
    """
    Simulates loading a synthetic, balanced dataset for training purposes.
    """
    
    NUM_SAMPLES_PER_CLASS = 12 
    
    X = []
    labels = []
    
    label_to_index = {name: i for i, name in enumerate(CLASS_NAMES)}
    
    for label_name in CLASS_NAMES:
        for _ in range(NUM_SAMPLES_PER_CLASS):
            # Generate a dummy image 
            dummy_img = np.random.rand(IMAGE_SIZE[0], IMAGE_SIZE[1], 3).astype(np.float32) * 255.0
            X.append(dummy_img)
            labels.append(label_name)

    X = np.array(X, dtype=np.float32) / 255.0 # Normalize dummy data
    label_indices = np.array([label_to_index[label] for label in labels])
    y = tf.keras.utils.to_categorical(label_indices, num_classes=NUM_CLASSES)
    
    return X, y

# --- 2. Model Definition and Training ---

@st.cache_resource
def build_and_train_model():
    """Builds, compiles, and trains the CNN model."""
    st.info("Simulating data loading and processing...")
    
    X, y = load_images_from_web_simulated()

    # Split the simulated dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=np.argmax(y, axis=1)
    )

    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', name='last_conv_layer'), # Target layer for Grad-CAM
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    st.info("Training CNN Model... (Using simulated data)")
    
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        verbose=0
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    try:
        report = classification_report(y_true_classes, y_pred_classes, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    except ValueError:
        report = {'accuracy': 0.0, 'macro avg': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0}}
    
    conf_mat = confusion_matrix(y_true_classes, y_pred_classes)

    return model, history, acc, report, conf_mat, X_test, y_test

# --- 3. Advanced Grad-CAM Implementation ---

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Computes and returns the Grad-CAM heatmap."""
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array[np.newaxis, ...]) 
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize heatmap
    if tf.reduce_max(heatmap) > 1e-8:
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    else:
        heatmap = tf.zeros_like(heatmap)
        
    heatmap = heatmap.numpy()

    return heatmap, preds[0].numpy()

def display_gradcam(img, heatmap, alpha=0.5):
    """Overlays the heatmap on the original image."""
    
    # Check for OpenCV (cv2) global availability
    if 'cv2' not in globals():
        return None, "OpenCV (cv2) is not available to generate the Grad-CAM visualization."

    heatmap = np.uint8(255 * heatmap)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
    resized_heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    colormap = cv2.COLORMAP_JET
    heatmap_jet = cv2.applyColorMap(resized_heatmap, colormap)

    superimposed_img_bgr = cv2.addWeighted(img_bgr, 1.0 - alpha, heatmap_jet, alpha, 0)
    superimposed_img_rgb = cv2.cvtColor(superimposed_img_bgr, cv2.COLOR_BGR2RGB)

    # Encode image to bytes for Streamlit display
    is_success, buffer = cv2.imencode(".png", cv2.cvtColor(superimposed_img_rgb, cv2.COLOR_BGR2RGB))
    if is_success:
        return buffer.tobytes(), None
    return None, "Failed to encode Grad-CAM image."

def process_uploaded_file(uploaded_file, model):
    """Handles uploaded file, runs prediction, and generates Grad-CAM."""
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
    if 'cv2' not in globals():
        st.error("Cannot proceed: OpenCV (cv2) is required for image processing.")
        return "N/A", 0.0, None

    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # Ensure image is not None (decoding failed)
    if img_bgr is None:
        st.error("Uploaded file could not be decoded as an image. Please check the file format.")
        return "N/A", 0.0, None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img_model_input = cv2.resize(img_rgb, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    img_array = img_model_input.astype('float32') / 255.0

    LAST_CONV_LAYER_NAME = 'last_conv_layer'
    heatmap, preds = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER_NAME)

    predicted_class_index = np.argmax(preds)
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = preds[predicted_class_index] * 100

    gradcam_img_bytes, error = display_gradcam(img_rgb, heatmap, alpha=0.5)
    
    if error:
        st.error(error)
        return predicted_class, confidence, None

    return predicted_class, confidence, gradcam_img_bytes

# --- 4. Streamlit Application (Dashboard) ---

def main():
    st.set_page_config(layout="wide", page_title="Advanced Cacao Disease Detector")

    st.title("🌿 Cacao Disease Detector & Explainable AI Dashboard")
    st.markdown("Upload an image of your cacao plant or pod to detect common diseases (or confirm health) and see **where** the model is looking with Grad-CAM.")
    st.info("⚠️ This model is trained on **simulated data** for demonstration. For real use, you must supply your own large, labeled image dataset.")
    
    # --- Sidebar for Model Metrics ---
    with st.sidebar:
        st.title("Model Training & Metrics")
        st.info(f"Model trained on **{NUM_CLASSES}** classes: {', '.join(CLASS_NAMES)}")

        with st.spinner("Building and training the model..."):
            # Load and train the model (cached)
            model, history, acc, report, conf_mat, X_test, y_test = build_and_train_model()
            st.success("Model trained successfully!")

        st.header("Model Validation Results (Simulated)")
        st.markdown(f"**Test Set Accuracy:** **{acc:.2f}**")
        
        if history.history:
            st.markdown(f"**Test Loss:** **{history.history['val_loss'][-1]:.4f}**")

        with st.expander("Detailed Classification Report"):
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)

        if SEABORN_AVAILABLE:
            with st.expander("Confusion Matrix"):
                fig, ax = plt.subplots(figsize=(6, 6))
                
                y_pred_classes_cmat = np.argmax(model.predict(X_test, verbose=0), axis=1)
                y_true_classes_cmat = np.argmax(y_test, axis=1)

                cmat_disp = confusion_matrix(y_true_classes_cmat, y_pred_classes_cmat)
                
                sns.heatmap(cmat_disp, annot=True, fmt='d', cmap='Blues', cbar=False,
                            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                ax.set_title('Confusion Matrix (Simulated)')
                st.pyplot(fig)
        else:
            st.warning("Seaborn is required to display the Confusion Matrix plot.")

    # --- Main App for Prediction ---
    st.header("Upload Cacao Image for Analysis")

    # Image Upload Widget (The user dashboard component)
    uploaded_file = st.file_uploader(
        "Upload an image of a cacao leaf or pod (JPEG/PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        st.subheader("Analysis Results")
        
        # Create responsive columns for the results display
        col_img, col_gradcam, col_description = st.columns([1, 1, 1.5]) 

        # Process and Predict
        predicted_class, confidence, gradcam_img_bytes = process_uploaded_file(uploaded_file, model)
        
        # COLUMN 1: Original Image and Prediction
        with col_img:
            st.markdown("<p style='text-align: center; font-size: 1.25rem; font-weight: bold;'>Your Uploaded Image</p>", unsafe_allow_html=True)
            st.image(uploaded_file, use_column_width=True)
            st.markdown(f"**Predicted Status:** <span style='color: #1E88E5; font-size: 1.5rem;'>**{predicted_class}**</span>", unsafe_allow_html=True)
            st.markdown(f"**Confidence:** <span style='color: #4CAF50; font-size: 1.5rem;'>**{confidence:.2f}%**</span>", unsafe_allow_html=True)

        # COLUMN 2: Grad-CAM Visualization
        with col_gradcam:
            st.markdown("<p style='text-align: center; font-size: 1.25rem; font-weight: bold;'>Model Focus (Grad-CAM)</p>", unsafe_allow_html=True)
            if gradcam_img_bytes:
                base64_img = base64.b64encode(gradcam_img_bytes).decode('utf-8')
                st.image(f"data:image/png;base64,{base64_img}", use_column_width=True)
                st.markdown('''
                <small>The **red/yellow areas** indicate where the model focused its attention to make the prediction. This acts like an **arrow pointing to the key visual evidence** for the detected disease or health.</small>
                ''', unsafe_allow_html=True)
            else:
                st.error("Could not generate Grad-CAM visualization. Ensure OpenCV is correctly installed.")
        
        # COLUMN 3: Disease Description
        with col_description:
            st.markdown("<p style='text-align: center; font-size: 1.25rem; font-weight: bold;'>Detailed Information</p>", unsafe_allow_html=True)
            if predicted_class in DISEASE_DESCRIPTIONS:
                disease_info = DISEASE_DESCRIPTIONS[predicted_class]
                st.markdown(f"### {predicted_class.replace('_', ' ')}")
                st.markdown(f"**Description:** {disease_info.get('description', 'No detailed description available.')}")
                if disease_info.get('causes'):
                    st.markdown(f"**Causes:** {disease_info['causes']}")
                if disease_info.get('treatment'):
                    st.markdown(f"**Treatment/Management:** {disease_info['treatment']}")
                if disease_info.get('severity'):
                    st.markdown(f"**Severity:** {disease_info['severity']}")
            else:
                st.info("No specific description available for this prediction.")

    else:
        st.info("Please upload an image of a cacao plant or pod to get started!")

if __name__ == '__main__':
    main()EPOCHS = 15 

# Placeholder for disease descriptions
DISEASE_DESCRIPTIONS = {
    "Healthy": {
        "description": "The cacao plant shows no signs of disease and appears to be thriving. Healthy leaves are typically green, vibrant, and free from spots or discoloration.",
        "causes": "Proper nutrition, adequate water, good sunlight, and protection from pests contribute to a healthy plant.",
        "treatment": "Maintain optimal growing conditions."
    },
    "Witches_Broom": {
        "description": "Witches' Broom disease, caused by the fungus Moniliophthora perniciosa, leads to abnormal, dense growths of shoots or 'brooms' and swollen stems. It can also affect pods, causing premature ripening and malformation.",
        "causes": "Fungal infection (Moniliophthora perniciosa).",
        "treatment": "Pruning infected parts, resistant varieties, fungicides. Strict phytosanitary measures are crucial.",
        "severity": "High - can significantly reduce yield and lead to tree death if untreated."
    },
    "Black_Pod_Rot": {
        "description": "Black Pod Rot, caused by various Phytophthora species, is characterized by rapidly spreading black or brown lesions on cacao pods, eventually covering the entire pod. It also affects flowers, cushions, and leaves.",
        "causes": "Fungal-like oomycete infection (Phytophthora spp.), favored by high humidity and rainfall.",
        "treatment": "Sanitation (removing infected pods), copper-based fungicides, good drainage, wider spacing of trees. Breeding for resistance is ongoing.",
        "severity": "High - major cause of yield loss globally."
    },
    "Frosty_Pod_Rot": {
        "description": "Frosty Pod Rot, caused by the fungus Moniliophthora roreri, initially appears as small, water-soaked spots on pods, which then develop a white, powdery fungal growth (like frost). The internal beans turn black and rot, making the pods commercially worthless.",
        "causes": "Fungal infection (Moniliophthora roreri), spread by wind and rain.",
        "treatment": "Strict sanitation (removing mummified pods), shade management, pruning, and developing resistant varieties. Fungicides are generally less effective once symptoms appear.",
        "severity": "High - causes severe economic losses in affected regions."
    }
} # FINAL CLOSING BRACE FOR DICTIONARY


@st.cache_data
def load_images_from_web_simulated():
    """
    Simulates loading a synthetic, balanced dataset for training purposes.
    This ensures the app runs and demonstrates the classification/Grad-CAM logic.
    """
    
    NUM_SAMPLES_PER_CLASS = 12 
    
    X = []
    labels = []
    
    label_to_index = {name: i for i, name in enumerate(CLASS_NAMES)}
    
    for label_name in CLASS_NAMES:
        for _ in range(NUM_SAMPLES_PER_CLASS):
            # Generate a dummy image 
            dummy_img = np.random.rand(IMAGE_SIZE[0], IMAGE_SIZE[1], 3).astype(np.float32) * 255.0
            X.append(dummy_img)
            labels.append(label_name)

    X = np.array(X, dtype=np.float32) / 255.0 # Normalize dummy data
    label_indices = np.array([label_to_index[label] for label in labels])
    y = tf.keras.utils.to_categorical(label_indices, num_classes=NUM_CLASSES)
    
    return X, y

# --- 2. Model Definition and Training ---

@st.cache_resource
def build_and_train_model():
    """Builds, compiles, and trains the CNN model."""
    st.info("Simulating data loading and processing...")
    
    X, y = load_images_from_web_simulated()

    # Split the simulated dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=np.argmax(y, axis=1)
    )

    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', name='last_conv_layer'), # Target layer for Grad-CAM
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    st.info("Training CNN Model... (Using simulated data)")
    
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        verbose=0
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    try:
        report = classification_report(y_true_classes, y_pred_classes, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    except ValueError:
        report = {'accuracy': 0.0, 'macro avg': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0}}
    
    conf_mat = confusion_matrix(y_true_classes, y_pred_classes)

    return model, history, acc, report, conf_mat, X_test, y_test

# --- 3. Advanced Grad-CAM Implementation ---

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Computes and returns the Grad-CAM heatmap."""
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array[np.newaxis, ...]) 
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize heatmap
    if tf.reduce_max(heatmap) > 1e-8:
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    else:
        heatmap = tf.zeros_like(heatmap)
        
    heatmap = heatmap.numpy()

    return heatmap, preds[0].numpy()

def display_gradcam(img, heatmap, alpha=0.5):
    """Overlays the heatmap on the original image."""
    
    # Check for OpenCV (cv2) global availability
    if 'cv2' not in globals():
        return None, "OpenCV (cv2) is not available to generate the Grad-CAM visualization."

    heatmap = np.uint8(255 * heatmap)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
    resized_heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    colormap = cv2.COLORMAP_JET
    heatmap_jet = cv2.applyColorMap(resized_heatmap, colormap)

    superimposed_img_bgr = cv2.addWeighted(img_bgr, 1.0 - alpha, heatmap_jet, alpha, 0)
    superimposed_img_rgb = cv2.cvtColor(superimposed_img_bgr, cv2.COLOR_BGR2RGB)

    # Encode image to bytes for Streamlit display
    is_success, buffer = cv2.imencode(".png", cv2.cvtColor(superimposed_img_rgb, cv2.COLOR_BGR2RGB))
    if is_success:
        return buffer.tobytes(), None
    return None, "Failed to encode Grad-CAM image."

def process_uploaded_file(uploaded_file, model):
    """Handles uploaded file, runs prediction, and generates Grad-CAM."""
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
    if 'cv2' not in globals():
        st.error("Cannot proceed: OpenCV (cv2) is required for image processing.")
        return "N/A", 0.0, None

    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # Ensure image is not None (decoding failed)
    if img_bgr is None:
        st.error("Uploaded file could not be decoded as an image. Please check the file format.")
        return "N/A", 0.0, None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img_model_input = cv2.resize(img_rgb, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    img_array = img_model_input.astype('float32') / 255.0

    LAST_CONV_LAYER_NAME = 'last_conv_layer'
    heatmap, preds = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER_NAME)

    predicted_class_index = np.argmax(preds)
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = preds[predicted_class_index] * 100

    gradcam_img_bytes, error = display_gradcam(img_rgb, heatmap, alpha=0.5)
    
    if error:
        st.error(error)
        return predicted_class, confidence, None

    return predicted_class, confidence, gradcam_img_bytes

# --- 4. Streamlit Application (Dashboard) ---

def main():
    st.set_page_config(layout="wide", page_title="Advanced Cacao Disease Detector")

    st.title("🌿 Cacao Disease Detector & Explainable AI Dashboard")
    st.markdown("Upload an image of your cacao plant or pod to detect common diseases (or confirm health) and see **where** the model is looking with Grad-CAM.")
    st.info("⚠️ This model is trained on **simulated data** for demonstration. For real use, you must supply your own large, labeled image dataset.")
    
    # --- Sidebar for Model Metrics ---
    with st.sidebar:
        st.title("Model Training & Metrics")
        st.info(f"Model trained on **{NUM_CLASSES}** classes: {', '.join(CLASS_NAMES)}")

        with st.spinner("Building and training the model..."):
            # Load and train the model (cached)
            model, history, acc, report, conf_mat, X_test, y_test = build_and_train_model()
            st.success("Model trained successfully!")

        st.header("Model Validation Results (Simulated)")
        st.markdown(f"**Test Set Accuracy:** **{acc:.2f}**")
        
        if history.history:
            st.markdown(f"**Test Loss:** **{history.history['val_loss'][-1]:.4f}**")

        with st.expander("Detailed Classification Report"):
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)

        if SEABORN_AVAILABLE:
            with st.expander("Confusion Matrix"):
                fig, ax = plt.subplots(figsize=(6, 6))
                
                y_pred_classes_cmat = np.argmax(model.predict(X_test, verbose=0), axis=1)
                y_true_classes_cmat = np.argmax(y_test, axis=1)

                cmat_disp = confusion_matrix(y_true_classes_cmat, y_pred_classes_cmat)
                
                sns.heatmap(cmat_disp, annot=True, fmt='d', cmap='Blues', cbar=False,
                            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                ax.set_title('Confusion Matrix (Simulated)')
                st.pyplot(fig)
        else:
            st.warning("Seaborn is required to display the Confusion Matrix plot.")

    # --- Main App for Prediction ---
    st.header("Upload Cacao Image for Analysis")

    # Image Upload Widget (The user dashboard component)
    uploaded_file = st.file_uploader(
        "Upload an image of a cacao leaf or pod (JPEG/PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        st.subheader("Analysis Results")
        
        # Create responsive columns for the results display
        col_img, col_gradcam, col_description = st.columns([1, 1, 1.5]) 

        # Process and Predict
        predicted_class, confidence, gradcam_img_bytes = process_uploaded_file(uploaded_file, model)
        
        # COLUMN 1: Original Image and Prediction
        with col_img:
            st.markdown("<p style='text-align: center; font-size: 1.25rem; font-weight: bold;'>Your Uploaded Image</p>", unsafe_allow_html=True)
            st.image(uploaded_file, use_column_width=True)
            st.markdown(f"**Predicted Status:** <span style='color: #1E88E5; font-size: 1.5rem;'>**{predicted_class}**</span>", unsafe_allow_html=True)
            st.markdown(f"**Confidence:** <span style='color: #4CAF50; font-size: 1.5rem;'>**{confidence:.2f}%**</span>", unsafe_allow_html=True)

        # COLUMN 2: Grad-CAM Visualization
        with col_gradcam:
            st.markdown("<p style='text-align: center; font-size: 1.25rem; font-weight: bold;'>Model Focus (Grad-CAM)</p>", unsafe_allow_html=True)
            if gradcam_img_bytes:
                base64_img = base64.b64encode(gradcam_img_bytes).decode('utf-8')
                st.image(f"data:image/png;base64,{base64_img}", use_column_width=True)
                st.markdown('''
                <small>The **red/yellow areas** indicate where the model focused its attention to make the prediction. This acts like an **arrow pointing to the key visual evidence** for the detected disease or health.</small>
                ''', unsafe_allow_html=True)
            else:
                st.error("Could not generate Grad-CAM visualization. Ensure OpenCV is correctly installed.")
        
        # COLUMN 3: Disease Description
        with col_description:
            st.markdown("<p style='text-align: center; font-size: 1.25rem; font-weight: bold;'>Detailed Information</p>", unsafe_allow_html=True)
            if predicted_class in DISEASE_DESCRIPTIONS:
                disease_info = DISEASE_DESCRIPTIONS[predicted_class]
                st.markdown(f"### {predicted_class.replace('_', ' ')}")
                st.markdown(f"**Description:** {disease_info.get('description', 'No detailed description available.')}")
                if disease_info.get('causes'):
                    st.markdown(f"**Causes:** {disease_info['causes']}")
                if disease_info.get('treatment'):
                    st.markdown(f"**Treatment/Management:** {disease_info['treatment']}")
                if disease_info.get('severity'):
                    st.markdown(f"**Severity:** {disease_info['severity']}")
            else:
                st.info("No specific description available for this prediction.")

    else:
        st.info("Please upload an image of a cacao plant or pod to get started!")

if __name__ == '__main__':
    main()EPOCHS = 15 

# Placeholder for disease descriptions
DISEASE_DESCRIPTIONS = {
    "Healthy": {
        "description": "The cacao plant shows no signs of disease and appears to be thriving. Healthy leaves are typically green, vibrant, and free from spots or discoloration.",
        "causes": "Proper nutrition, adequate water, good sunlight, and protection from pests contribute to a healthy plant.",
        "treatment": "Maintain optimal growing conditions."
    },
    "Witches_Broom": {
        "description": "Witches' Broom disease, caused by the fungus Moniliophthora perniciosa, leads to abnormal, dense growths of shoots or 'brooms' and swollen stems. It can also affect pods, causing premature ripening and malformation.",
        "causes": "Fungal infection (Moniliophthora perniciosa).",
        "treatment": "Pruning infected parts, resistant varieties, fungicides. Strict phytosanitary measures are crucial.",
        "severity": "High - can significantly reduce yield and lead to tree death if untreated."
    },
    "Black_Pod_Rot": {
        "description": "Black Pod Rot, caused by various Phytophthora species, is characterized by rapidly spreading black or brown lesions on cacao pods, eventually covering the entire pod. It also affects flowers, cushions, and leaves.",
        "causes": "Fungal-like oomycete infection (Phytophthora spp.), favored by high humidity and rainfall.",
        "treatment": "Sanitation (removing infected pods), copper-based fungicides, good drainage, wider spacing of trees. Breeding for resistance is ongoing.",
        "severity": "High - major cause of yield loss globally."
    },
    "Frosty_Pod_Rot": {
        "description": "Frosty Pod Rot, caused by the fungus Moniliophthora roreri, initially appears as small, water-soaked spots on pods, which then develop a white, powdery fungal growth (like frost). The internal beans turn black and rot, making the pods commercially worthless.",
        "causes": "Fungal infection (Moniliophthora roreri), spread by wind and rain.",
        "treatment": "Strict sanitation (removing mummified pods), shade management, pruning, and developing resistant varieties. Fungicides are generally less effective once symptoms appear.",
        "severity": "High - causes severe economic losses in affected regions."
    }
}


@st.cache_data
def load_images_from_web_simulated():
    """
    Simulates loading a synthetic, balanced dataset for training purposes.
    This ensures the app runs and demonstrates the classification/Grad-CAM logic.
    """
    
    NUM_SAMPLES_PER_CLASS = 12 
    
    X = []
    labels = []
    
    label_to_index = {name: i for i, name in enumerate(CLASS_NAMES)}
    
    for label_name in CLASS_NAMES:
        for _ in range(NUM_SAMPLES_PER_CLASS):
            # Generate a dummy image 
            dummy_img = np.random.rand(IMAGE_SIZE[0], IMAGE_SIZE[1], 3).astype(np.float32) * 255.0
            X.append(dummy_img)
            labels.append(label_name)

    X = np.array(X, dtype=np.float32) / 255.0 # Normalize dummy data
    label_indices = np.array([label_to_index[label] for label in labels])
    y = tf.keras.utils.to_categorical(label_indices, num_classes=NUM_CLASSES)
    
    return X, y

# --- 2. Model Definition and Training ---

@st.cache_resource
def build_and_train_model():
    """Builds, compiles, and trains the CNN model."""
    st.info("Simulating data loading and processing...")
    
    X, y = load_images_from_web_simulated()

    # Split the simulated dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=np.argmax(y, axis=1)
    )

    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', name='last_conv_layer'), # Target layer for Grad-CAM
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    st.info("Training CNN Model... (Using simulated data)")
    
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        verbose=0
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    try:
        report = classification_report(y_true_classes, y_pred_classes, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    except ValueError:
        report = {'accuracy': 0.0, 'macro avg': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0}}
    
    conf_mat = confusion_matrix(y_true_classes, y_pred_classes)

    return model, history, acc, report, conf_mat, X_test, y_test

# --- 3. Advanced Grad-CAM Implementation ---

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Computes and returns the Grad-CAM heatmap."""
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array[np.newaxis, ...]) 
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize heatmap
    if tf.reduce_max(heatmap) > 1e-8:
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    else:
        heatmap = tf.zeros_like(heatmap)
        
    heatmap = heatmap.numpy()

    return heatmap, preds[0].numpy()

def display_gradcam(img, heatmap, alpha=0.5):
    """Overlays the heatmap on the original image."""
    
    # Check for OpenCV (cv2) global availability
    if 'cv2' not in globals():
        return None, "OpenCV (cv2) is not available to generate the Grad-CAM visualization."

    heatmap = np.uint8(255 * heatmap)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
    resized_heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    colormap = cv2.COLORMAP_JET
    heatmap_jet = cv2.applyColorMap(resized_heatmap, colormap)

    superimposed_img_bgr = cv2.addWeighted(img_bgr, 1.0 - alpha, heatmap_jet, alpha, 0)
    superimposed_img_rgb = cv2.cvtColor(superimposed_img_bgr, cv2.COLOR_BGR2RGB)

    # Encode image to bytes for Streamlit display
    is_success, buffer = cv2.imencode(".png", cv2.cvtColor(superimposed_img_rgb, cv2.COLOR_BGR2RGB))
    if is_success:
        return buffer.tobytes(), None
    return None, "Failed to encode Grad-CAM image."

def process_uploaded_file(uploaded_file, model):
    """Handles uploaded file, runs prediction, and generates Grad-CAM."""
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
    if 'cv2' not in globals():
        st.error("Cannot proceed: OpenCV (cv2) is required for image processing.")
        return "N/A", 0.0, None

    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # Ensure image is not None (decoding failed)
    if img_bgr is None:
        st.error("Uploaded file could not be decoded as an image. Please check the file format.")
        return "N/A", 0.0, None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img_model_input = cv2.resize(img_rgb, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    img_array = img_model_input.astype('float32') / 255.0

    LAST_CONV_LAYER_NAME = 'last_conv_layer'
    heatmap, preds = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER_NAME)

    predicted_class_index = np.argmax(preds)
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = preds[predicted_class_index] * 100

    gradcam_img_bytes, error = display_gradcam(img_rgb, heatmap, alpha=0.5)
    
    if error:
        st.error(error)
        return predicted_class, confidence, None

    return predicted_class, confidence, gradcam_img_bytes

# --- 4. Streamlit Application (Dashboard) ---

def main():
    st.set_page_config(layout="wide", page_title="Advanced Cacao Disease Detector")

    st.title("🌿 Cacao Disease Detector & Explainable AI Dashboard")
    st.markdown("Upload an image of your cacao plant or pod to detect common diseases (or confirm health) and see **where** the model is looking with Grad-CAM.")
    st.info("⚠️ This model is trained on **simulated data** for demonstration. For real use, you must supply your own large, labeled image dataset.")
    
    # --- Sidebar for Model Metrics ---
    with st.sidebar:
        st.title("Model Training & Metrics")
        st.info(f"Model trained on **{NUM_CLASSES}** classes: {', '.join(CLASS_NAMES)}")

        with st.spinner("Building and training the model..."):
            # Load and train the model (cached)
            model, history, acc, report, conf_mat, X_test, y_test = build_and_train_model()
            st.success("Model trained successfully!")

        st.header("Model Validation Results (Simulated)")
        st.markdown(f"**Test Set Accuracy:** **{acc:.2f}**")
        
        if history.history:
            st.markdown(f"**Test Loss:** **{history.history['val_loss'][-1]:.4f}**")

        with st.expander("Detailed Classification Report"):
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)

        if SEABORN_AVAILABLE:
            with st.expander("Confusion Matrix"):
                fig, ax = plt.subplots(figsize=(6, 6))
                
                y_pred_classes_cmat = np.argmax(model.predict(X_test, verbose=0), axis=1)
                y_true_classes_cmat = np.argmax(y_test, axis=1)

                cmat_disp = confusion_matrix(y_true_classes_cmat, y_pred_classes_cmat)
                
                sns.heatmap(cmat_disp, annot=True, fmt='d', cmap='Blues', cbar=False,
                            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                ax.set_title('Confusion Matrix (Simulated)')
                st.pyplot(fig)
        else:
            st.warning("Seaborn is required to display the Confusion Matrix plot.")

    # --- Main App for Prediction ---
    st.header("Upload Cacao Image for Analysis")

    # Image Upload Widget (The user dashboard component)
    uploaded_file = st.file_uploader(
        "Upload an image of a cacao leaf or pod (JPEG/PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        st.subheader("Analysis Results")
        
        # Create responsive columns for the results display
        col_img, col_gradcam, col_description = st.columns([1, 1, 1.5]) 

        # Process and Predict
        predicted_class, confidence, gradcam_img_bytes = process_uploaded_file(uploaded_file, model)
        
        # COLUMN 1: Original Image and Prediction
        with col_img:
            st.markdown("<p style='text-align: center; font-size: 1.25rem; font-weight: bold;'>Your Uploaded Image</p>", unsafe_allow_html=True)
            st.image(uploaded_file, use_column_width=True)
            st.markdown(f"**Predicted Status:** <span style='color: #1E88E5; font-size: 1.5rem;'>**{predicted_class}**</span>", unsafe_allow_html=True)
            st.markdown(f"**Confidence:** <span style='color: #4CAF50; font-size: 1.5rem;'>**{confidence:.2f}%**</span>", unsafe_allow_html=True)

        # COLUMN 2: Grad-CAM Visualization
        with col_gradcam:
            st.markdown("<p style='text-align: center; font-size: 1.25rem; font-weight: bold;'>Model Focus (Grad-CAM)</p>", unsafe_allow_html=True)
            if gradcam_img_bytes:
                base64_img = base64.b64encode(gradcam_img_bytes).decode('utf-8')
                st.image(f"data:image/png;base64,{base64_img}", use_column_width=True)
                st.markdown('''
                <small>The **red/yellow areas** indicate where the model focused its attention to make the prediction. This acts like an **arrow pointing to the key visual evidence** for the detected disease or health.</small>
                ''', unsafe_allow_html=True)
            else:
                st.error("Could not generate Grad-CAM visualization. Ensure OpenCV is correctly installed.")
        
        # COLUMN 3: Disease Description
        with col_description:
            st.markdown("<p style='text-align: center; font-size: 1.25rem; font-weight: bold;'>Detailed Information</p>", unsafe_allow_html=True)
            if predicted_class in DISEASE_DESCRIPTIONS:
                disease_info = DISEASE_DESCRIPTIONS[predicted_class]
                st.markdown(f"### {predicted_class.replace('_', ' ')}")
                st.markdown(f"**Description:** {disease_info.get('description', 'No detailed description available.')}")
                if disease_info.get('causes'):
                    st.markdown(f"**Causes:** {disease_info['causes']}")
                if disease_info.get('treatment'):
                    st.markdown(f"**Treatment/Management:** {disease_info['treatment']}")
                if disease_info.get('severity'):
                    st.markdown(f"**Severity:** {disease_info['severity']}")
            else:
                st.info("No specific description available for this prediction.")

    else:
        st.info("Please upload an image of a cacao plant or pod to get started!")

if __name__ == '__main__':
    main()EPOCHS = 15 # Increased epochs slightly for potentially better (simulated) learning

# Placeholder for disease descriptions (no structural changes needed here, as the previous errors were elsewhere)
DISEASE_DESCRIPTIONS = {
    "Healthy": {
        "description": "The cacao plant shows no signs of disease and appears to be thriving. Healthy leaves are typically green, vibrant, and free from spots or discoloration.",
        "causes": "Proper nutrition, adequate water, good sunlight, and protection from pests contribute to a healthy plant.",
        "treatment": "Maintain optimal growing conditions."
    },
    "Witches_Broom": {
        "description": "Witches' Broom disease, caused by the fungus Moniliophthora perniciosa, leads to abnormal, dense growths of shoots or 'brooms' and swollen stems. It can also affect pods, causing premature ripening and malformation.",
        "causes": "Fungal infection (Moniliophthora perniciosa).",
        "treatment": "Pruning infected parts, resistant varieties, fungicides. Strict phytosanitary measures are crucial.",
        "severity": "High - can significantly reduce yield and lead to tree death if untreated."
    },
    "Black_Pod_Rot": {
        "description": "Black Pod Rot, caused by various Phytophthora species, is characterized by rapidly spreading black or brown lesions on cacao pods, eventually covering the entire pod. It also affects flowers, cushions, and leaves.",
        "causes": "Fungal-like oomycete infection (Phytophthora spp.), favored by high humidity and rainfall.",
        "treatment": "Sanitation (removing infected pods), copper-based fungicides, good drainage, wider spacing of trees. Breeding for resistance is ongoing.",
        "severity": "High - major cause of yield loss globally."
    },
    "Frosty_Pod_Rot": {
        "description": "Frosty Pod Rot, caused by the fungus Moniliophthora roreri, initially appears as small, water-soaked spots on pods, which then develop a white, powdery fungal growth (like frost). The internal beans turn black and rot, making the pods commercially worthless.",
        "causes": "Fungal infection (Moniliophthora roreri), spread by wind and rain.",
        "treatment": "Strict sanitation (removing mummified pods), shade management, pruning, and developing resistant varieties. Fungicides are generally less effective once symptoms appear.",
        "severity": "High - causes severe economic losses in affected regions."
    }
}


@st.cache_data
def load_images_from_web_simulated():
    """
    Simulates loading images from a web source for training purposes.
    
    ***IMPORTANT: For real deployment, you must replace the dummy image generation
    with actual code to fetch images from PUBLIC, DIRECT web URLs.***
    """
    
    # We simulate a balanced dataset for demonstration purposes
    NUM_SAMPLES_PER_CLASS = 12 
    TOTAL_SAMPLES = NUM_CLASSES * NUM_SAMPLES_PER_CLASS
    
    X = []
    labels = []
    
    label_to_index = {name: i for i, name in enumerate(CLASS_NAMES)}
    
    for label_name in CLASS_NAMES:
        for _ in range(NUM_SAMPLES_PER_CLASS):
            # Generate a dummy image (simulating a downloaded and processed image)
            dummy_img = np.random.rand(IMAGE_SIZE[0], IMAGE_SIZE[1], 3).astype(np.float32) * 255.0
            X.append(dummy_img)
            labels.append(label_name)

    X = np.array(X, dtype=np.float32) / 255.0 # Normalize dummy data
    label_indices = np.array([label_to_index[label] for label in labels])
    y = tf.keras.utils.to_categorical(label_indices, num_classes=NUM_CLASSES)
    
    return X, y

# --- 2. Model Definition and Training ---

@st.cache_resource
def build_and_train_model():
    """Builds, compiles, and trains the CNN model."""
    st.info("Simulating data loading from a web source and processing...")
    
    X, y = load_images_from_web_simulated()

    # Split the simulated dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=np.argmax(y, axis=1)
    )

    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', name='last_conv_layer'), # Target layer for Grad-CAM
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    st.info("Training CNN Model... (Using simulated web-sourced data)")
    
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        verbose=0
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    try:
        report = classification_report(y_true_classes, y_pred_classes, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    except ValueError:
        report = {'accuracy': 0.0, 'macro avg': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0}}
    
    conf_mat = confusion_matrix(y_true_classes, y_pred_classes)

    return model, history, acc, report, conf_mat, X_test, y_test

# --- 3. Advanced Grad-CAM Implementation ---

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Computes and returns the Grad-CAM heatmap.
    """
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array[np.newaxis, ...]) 
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Ensure heatmap is not all zeros before division
    if tf.reduce_max(heatmap) > 1e-8:
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    else:
        heatmap = tf.zeros_like(heatmap) # Return all zeros if no activation
        
    heatmap = heatmap.numpy()

    return heatmap, preds[0].numpy()

def display_gradcam(img, heatmap, alpha=0.5):
    """
    Overlays the heatmap on the original image and returns a new image (as bytes).
    """
    if 'cv2' not in globals():
        return None, "OpenCV (cv2) is not available to generate the Grad-CAM visualization."

    heatmap = np.uint8(255 * heatmap)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
    resized_heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    colormap = cv2.COLORMAP_JET
    heatmap_jet = cv2.applyColorMap(resized_heatmap, colormap)

    superimposed_img_bgr = cv2.addWeighted(img_bgr, 1.0 - alpha, heatmap_jet, alpha, 0)

    superimposed_img_rgb = cv2.cvtColor(superimposed_img_bgr, cv2.COLOR_BGR2RGB)

    is_success, buffer = cv2.imencode(".png", cv2.cvtColor(superimposed_img_rgb, cv2.COLOR_BGR2RGB))
    if is_success:
        return buffer.tobytes(), None
    return None, "Failed to encode Grad-CAM image."

def process_uploaded_file(uploaded_file, model):
    """Handles uploaded file, runs prediction, and generates Grad-CAM."""
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
    if 'cv2' not in globals():
        st.error("Cannot proceed: OpenCV (cv2) is required for image processing.")
        return "N/A", 0.0, None

    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # Ensure image is not None (decoding failed)
    if img_bgr is None:
        st.error("Uploaded file could not be decoded as an image. Please check the file format.")
        return "N/A", 0.0, None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img_model_input = cv2.resize(img_rgb, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    img_array = img_model_input.astype('float32') / 255.0

    LAST_CONV_LAYER_NAME = 'last_conv_layer'
    heatmap, preds = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER_NAME)

    predicted_class_index = np.argmax(preds)
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = preds[predicted_class_index] * 100

    gradcam_img_bytes, error = display_gradcam(img_rgb, heatmap, alpha=0.5)
    
    if error:
        st.error(error)
        return predicted_class, confidence, None

    return predicted_class, confidence, gradcam_img_bytes

# --- 4. Streamlit Application ---

def main():
    st.set_page_config(layout="wide", page_title="Advanced Cacao Disease Detector")

    st.title("🌿 Cacao Disease Detector & Explainable AI Dashboard")
    st.markdown("Upload an image of your cacao plant or pod to detect common diseases (or confirm health) and see *where* the model is looking with Grad-CAM.")
    st.info("The model is trained using a simulated dataset to demonstrate functionality. For real-world use, replace the dummy data loading with actual image fetching from a public web source.")
    st.warning("🚨 **IMPORTANT:** Direct Google Drive folder links cannot be used for data loading in deployed Streamlit apps. Please host your data publicly (e.g., on GitHub raw links or dedicated image hosting) and update the `load_images_from_web_simulated` function.")

    # --- Sidebar for Model Metrics ---
    with st.sidebar:
        st.title("Model Training & Metrics")
        st.info(f"Model trained on **{NUM_CLASSES}** classes: {', '.join(CLASS_NAMES)}")

        with st.spinner("Building and training the model..."):
            model, history, acc, report, conf_mat, X_test, y_test = build_and_train_model()
            st.success("Model trained successfully!")

        st.header("Model Validation Results (Simulated)")
        st.markdown(f"**Test Set Accuracy:** **{acc:.2f}**")
        
        if history.history:
            st.markdown(f"**Test Loss:** **{history.history['val_loss'][-1]:.4f}**")

        with st.expander("Detailed Classification Report"):
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)

        if SEABORN_AVAILABLE:
            with st.expander("Confusion Matrix"):
                fig, ax = plt.subplots(figsize=(6, 6))
                
                y_pred_classes_cmat = np.argmax(model.predict(X_test, verbose=0), axis=1)
                y_true_classes_cmat = np.argmax(y_test, axis=1)

                cmat_disp = confusion_matrix(y_true_classes_cmat, y_pred_classes_cmat)
                
                sns.heatmap(cmat_disp, annot=True, fmt='d', cmap='Blues', cbar=False,
                            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                ax.set_title('Confusion Matrix (Simulated)')
                st.pyplot(fig)
        else:
            st.warning("Seaborn is required to display the Confusion Matrix plot.")

    # --- Main App for Prediction ---
    st.header("Upload Cacao Image for Analysis")

    uploaded_file = st.file_uploader(
        "Upload an image of a cacao leaf or pod (JPEG/PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        st.subheader("Analysis Results")
        
        # Create columns for original image, Grad-CAM, and description
        col_img, col_gradcam, col_description = st.columns([1, 1, 1.5]) # Adjusted column width for description

        # Process and Predict
        predicted_class, confidence, gradcam_img_bytes = process_uploaded_file(uploaded_file, model)
        
        with col_img:
            st.markdown("<p style='text-align: center; font-size: 1.25rem; font-weight: bold;'>Your Uploaded Image</p>", unsafe_allow_html=True)
            st.image(uploaded_file, use_column_width=True)
            st.markdown(f"**Predicted Status:** <span style='color: #1E88E5; font-size: 1.5rem;'>**{predicted_class}**</span>", unsafe_allow_html=True)
            st.markdown(f"**Confidence:** <span style='color: #4CAF50; font-size: 1.5rem;'>**{confidence:.2f}%**</span>", unsafe_allow_html=True)

        with col_gradcam:
            st.markdown("<p style='text-align: center; font-size: 1.25rem; font-weight: bold;'>Model Focus (Grad-CAM)</p>", unsafe_allow_html=True)
            if gradcam_img_bytes:
                base64_img = base64.b64encode(gradcam_img_bytes).decode('utf-8')
                st.image(f"data:image/png;base64,{base64_img}", use_column_width=True)
                # Used triple single quotes for robustness against environment issues
                st.markdown('''
                <small>The **red/yellow areas** indicate where the model focused its attention to make the prediction. This acts like an **arrow pointing to the key visual evidence** for the detected disease or health.</small>
                ''', unsafe_allow_html=True)
            else:
                st.error("Could not generate Grad-CAM visualization. Ensure OpenCV is installed.")
        
        with col_description:
            st.markdown("<p style='text-align: center; font-size: 1.25rem; font-weight: bold;'>Detailed Information</p>", unsafe_allow_html=True)
            if predicted_class in DISEASE_DESCRIPTIONS:
                disease_info = DISEASE_DESCRIPTIONS[predicted_class]
                st.markdown(f"### {predicted_class.replace('_', ' ')}")
                st.markdown(f"**Description:** {disease_info.get('description', 'No detailed description available.')}")
                if disease_info.get('causes'):
                    st.markdown(f"**Causes:** {disease_info['causes']}")
                if disease_info.get('treatment'):
                    st.markdown(f"**Treatment/Management:** {disease_info['treatment']}")
                if disease_info.get('severity'):
                    st.markdown(f"**Severity:** {disease_info['severity']}")
            else:
                st.info("No specific description available for this prediction. Please refer to general cacao plant care.")

    else:
        st.info("Please upload an image of a cacao plant or pod to get started!")

if __name__ == '__main__':
    main()    {"filename": "frosty pod rot_00003.png", "label": "Frosty_Pod_Rot"},
    {"filename": "healthy_00003.png", "label": "Healthy"},
] 

CLASS_NAMES = sorted(list(set(item['label'] for item in IMAGE_FILE_DATA)))
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 4 # Reduced batch size due to very small dataset
EPOCHS = 15 # Increased epochs slightly for potentially better (simulated) learning


# Placeholder for disease descriptions
DISEASE_DESCRIPTIONS = {
    "Healthy": {
        "description": "The cacao plant shows no signs of disease and appears to be thriving. Healthy leaves are typically green, vibrant, and free from spots or discoloration.",
        "causes": "Proper nutrition, adequate water, good sunlight, and protection from pests contribute to a healthy plant.",
        "treatment": "Maintain optimal growing conditions."
    },
    "Witches_Broom": {
        "description": "Witches' Broom disease, caused by the fungus Moniliophthora perniciosa, leads to abnormal, dense growths of shoots or 'brooms' and swollen stems. It can also affect pods, causing premature ripening and malformation.",
        "causes": "Fungal infection (Moniliophthora perniciosa).",
        "treatment": "Pruning infected parts, resistant varieties, fungicides. Strict phytosanitary measures are crucial.",
        "severity": "High - can significantly reduce yield and lead to tree death if untreated."
    },
    "Black_Pod_Rot": {
        "description": "Black Pod Rot, caused by various Phytophthora species, is characterized by rapidly spreading black or brown lesions on cacao pods, eventually covering the entire pod. It also affects flowers, cushions, and leaves.",
        "causes": "Fungal-like oomycete infection (Phytophthora spp.), favored by high humidity and rainfall.",
        "treatment": "Sanitation (removing infected pods), copper-based fungicides, good drainage, wider spacing of trees. Breeding for resistance is ongoing.",
        "severity": "High - major cause of yield loss globally."
    },
    "Frosty_Pod_Rot": {
        "description": "Frosty Pod Rot, caused by the fungus Moniliophthora roreri, initially appears as small, water-soaked spots on pods, which then develop a white, powdery fungal growth (like frost). The internal beans turn black and rot, making the pods commercially worthless.",
        "causes": "Fungal infection (Moniliophthora roreri), spread by wind and rain.",
        "treatment": "Strict sanitation (removing mummified pods), shade management, pruning, and developing resistant varieties. Fungicides are generally less effective once symptoms appear.",
        "severity": "High - causes severe economic losses in affected regions."
    }
}


@st.cache_data
def load_images_from_web_simulated():
    """
    Simulates loading images from a web source for training purposes.
    
    ***IMPORTANT: For real deployment, you must replace the dummy image generation
    with actual code to fetch images from PUBLIC, DIRECT web URLs.
    Google Drive folder links are not directly accessible.***
    """
    
    X = []
    labels = []
    
    label_to_index = {name: i for i, name in enumerate(CLASS_NAMES)}
    
    for item in IMAGE_FILE_DATA:
        # --- PLACEHOLDER FOR REAL WEB FETCHING LOGIC ---
        # For simulation, we generate a dummy image:
        dummy_img = np.random.rand(IMAGE_SIZE[0], IMAGE_SIZE[1], 3).astype(np.float32) * 255.0
        X.append(dummy_img)
        labels.append(item['label'])

    X = np.array(X, dtype=np.float32) / 255.0 # Normalize dummy data
    label_indices = np.array([label_to_index[label] for label in labels])
    y = tf.keras.utils.to_categorical(label_indices, num_classes=NUM_CLASSES)
    
    return X, y

# --- 2. Model Definition and Training ---

@st.cache_resource
def build_and_train_model():
    """Builds, compiles, and trains the CNN model."""
    st.info("Simulating data loading from a web source and processing...")
    
    X, y = load_images_from_web_simulated()

    # Split the limited dataset, ensuring stratification for better class representation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=np.argmax(y, axis=1)
    )

    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', name='last_conv_layer'), # Target layer for Grad-CAM
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    st.info("Training CNN Model... (Using simulated web-sourced data)")
    
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        verbose=0
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    try:
        report = classification_report(y_true_classes, y_pred_classes, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    except ValueError:
        report = {'accuracy': 0.0, 'macro avg': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0}}
    
    conf_mat = confusion_matrix(y_true_classes, y_pred_classes)

    return model, history, acc, report, conf_mat, X_test, y_test

# --- 3. Advanced Grad-CAM Implementation ---

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Computes and returns the Grad-CAM heatmap.
    """
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array[np.newaxis, ...]) 
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Ensure heatmap is not all zeros before division
    if tf.reduce_max(heatmap) > 1e-8:
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    else:
        heatmap = tf.zeros_like(heatmap) # Return all zeros if no activation
        
    heatmap = heatmap.numpy()

    return heatmap, preds[0].numpy()

def display_gradcam(img, heatmap, alpha=0.5):
    """
    Overlays the heatmap on the original image and returns a new image (as bytes).
    """
    if 'cv2' not in globals():
        return None, "OpenCV (cv2) is not available to generate the Grad-CAM visualization."

    heatmap = np.uint8(255 * heatmap)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
    resized_heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    colormap = cv2.COLORMAP_JET
    heatmap_jet = cv2.applyColorMap(resized_heatmap, colormap)

    superimposed_img_bgr = cv2.addWeighted(img_bgr, 1.0 - alpha, heatmap_jet, alpha, 0)

    superimposed_img_rgb = cv2.cvtColor(superimposed_img_bgr, cv2.COLOR_BGR2RGB)

    is_success, buffer = cv2.imencode(".png", cv2.cvtColor(superimposed_img_rgb, cv2.COLOR_RGB2BGR))
    if is_success:
        return buffer.tobytes(), None
    return None, "Failed to encode Grad-CAM image."

def process_uploaded_image(uploaded_file, model):
    """Handles uploaded file, runs prediction, and generates Grad-CAM."""
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
    if 'cv2' not in globals():
        st.error("Cannot proceed: OpenCV (cv2) is required for image processing.")
        return "N/A", 0.0, None

    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # Ensure image is not None (decoding failed)
    if img_bgr is None:
        st.error("Uploaded file could not be decoded as an image. Please check the file format.")
        return "N/A", 0.0, None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img_model_input = cv2.resize(img_rgb, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    img_array = img_model_input.astype('float32') / 255.0

    LAST_CONV_LAYER_NAME = 'last_conv_layer'
    heatmap, preds = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER_NAME)

    predicted_class_index = np.argmax(preds)
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = preds[predicted_class_index] * 100

    gradcam_img_bytes, error = display_gradcam(img_rgb, heatmap, alpha=0.5)
    
    if error:
        st.error(error)
        return predicted_class, confidence, None

    return predicted_class, confidence, gradcam_img_bytes

# --- 4. Streamlit Application ---

def main():
    st.set_page_config(layout="wide", page_title="Advanced Cacao Disease Detector")

    st.title("🌿 Cacao Disease Detector & Explainable AI Dashboard")
    st.markdown("Upload an image of your cacao plant or pod to detect common diseases (or confirm health) and see *where* the model is looking with Grad-CAM.")
    st.info("The model is trained using a simulated dataset to demonstrate functionality. For real-world use, replace the dummy data loading with actual image fetching from a public web source.")
    st.warning("🚨 **IMPORTANT:** Direct Google Drive folder links cannot be used for data loading in deployed Streamlit apps. Please host your data publicly (e.g., on GitHub raw links or dedicated image hosting) and update the `load_images_from_web_simulated` function.")

    # --- Sidebar for Model Metrics ---
    with st.sidebar:
        st.title("Model Training & Metrics")
        st.info(f"Model trained on **{NUM_CLASSES}** classes: {', '.join(CLASS_NAMES)}")

        with st.spinner("Building and training the model..."):
            model, history, acc, report, conf_mat, X_test, y_test = build_and_train_model()
            st.success("Model trained successfully!")

        st.header("Model Validation Results (Simulated)")
        st.markdown(f"**Test Set Accuracy:** **{acc:.2f}**")
        
        if history.history:
            st.markdown(f"**Test Loss:** **{history.history['val_loss'][-1]:.4f}**")

        with st.expander("Detailed Classification Report"):
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)

        if SEABORN_AVAILABLE:
            with st.expander("Confusion Matrix"):
                fig, ax = plt.subplots(figsize=(6, 6))
                
                y_pred_classes_cmat = np.argmax(model.predict(X_test, verbose=0), axis=1)
                y_true_classes_cmat = np.argmax(y_test, axis=1)

                cmat_disp = confusion_matrix(y_true_classes_cmat, y_pred_classes_cmat)
                
                sns.heatmap(cmat_disp, annot=True, fmt='d', cmap='Blues', cbar=False,
                            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                ax.set_title('Confusion Matrix (Simulated)')
                st.pyplot(fig)
        else:
            st.warning("Seaborn is required to display the Confusion Matrix plot.")

    # --- Main App for Prediction ---
    st.header("Upload Cacao Image for Analysis")

    uploaded_file = st.file_uploader(
        "Upload an image of a cacao leaf or pod (JPEG/PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        st.subheader("Analysis Results")
        
        # Create columns for original image, Grad-CAM, and description
        col_img, col_gradcam, col_description = st.columns([1, 1, 1.5]) # Adjusted column width for description

        # Process and Predict
        predicted_class, confidence, gradcam_img_bytes = process_uploaded_file(uploaded_file, model)
        
        with col_img:
            st.markdown("<p style='text-align: center; font-size: 1.25rem; font-weight: bold;'>Your Uploaded Image</p>", unsafe_allow_html=True)
            st.image(uploaded_file, use_column_width=True)
            st.markdown(f"**Predicted Status:** <span style='color: #1E88E5; font-size: 1.5rem;'>**{predicted_class}**</span>", unsafe_allow_html=True)
            st.markdown(f"**Confidence:** <span style='color: #4CAF50; font-size: 1.5rem;'>**{confidence:.2f}%**</span>", unsafe_allow_html=True)

        with col_gradcam:
            st.markdown("<p style='text-align: center; font-size: 1.25rem; font-weight: bold;'>Model Focus (Grad-CAM)</p>", unsafe_allow_html=True)
            if gradcam_img_bytes:
                base64_img = base64.b64encode(gradcam_img_bytes).decode('utf-8')
                st.image(f"data:image/png;base64,{base64_img}", use_column_width=True)
                # Changed to triple single quotes for robustness
                st.markdown('''
                <small>The **red/yellow areas** indicate where the model focused its attention to make the prediction. This acts like an **arrow pointing to the key visual evidence** for the detected disease or health.</small>
                ''', unsafe_allow_html=True)
            else:
                st.error("Could not generate Grad-CAM visualization. Ensure OpenCV is installed.")
        
        with col_description:
            st.markdown("<p style='text-align: center; font-size: 1.25rem; font-weight: bold;'>Detailed Information</p>", unsafe_allow_html=True)
            if predicted_class in DISEASE_DESCRIPTIONS:
                disease_info = DISEASE_DESCRIPTIONS[predicted_class]
                st.markdown(f"### {predicted_class.replace('_', ' ')}")
                st.markdown(f"**Description:** {disease_info.get('description', 'No detailed description available.')}")
                if disease_info.get('causes'):
                    st.markdown(f"**Causes:** {disease_info['causes']}")
                if disease_info.get('treatment'):
                    st.markdown(f"**Treatment/Management:** {disease_info['treatment']}")
                if disease_info.get('severity'):
                    st.markdown(f"**Severity:** {disease_info['severity']}")
            else:
                st.info("No specific description available for this prediction. Please refer to general cacao plant care.")

    else:
        st.info("Please upload an image of a cacao plant or pod to get started!")

if __name__ == '__main__':
    main()    {"filename": "frosty pod rot_00003.png", "label": "Frosty_Pod_Rot"},
    {"filename": "healthy_00003.png", "label": "Healthy"},
] # This is the end of the list.

CLASS_NAMES = sorted(list(set(item['label'] for item in IMAGE_FILE_DATA)))
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 4 # Reduced batch size due to very small dataset
EPOCHS = 15 # Increased epochs slightly for potentially better (simulated) learning


# Placeholder for disease descriptions
DISEASE_DESCRIPTIONS = {
    "Healthy": {
        "description": "The cacao plant shows no signs of disease and appears to be thriving. Healthy leaves are typically green, vibrant, and free from spots or discoloration.",
        "causes": "Proper nutrition, adequate water, good sunlight, and protection from pests contribute to a healthy plant.",
        "treatment": "Maintain optimal growing conditions."
    },
    "Witches_Broom": {
        "description": "Witches' Broom disease, caused by the fungus Moniliophthora perniciosa, leads to abnormal, dense growths of shoots or 'brooms' and swollen stems. It can also affect pods, causing premature ripening and malformation.",
        "causes": "Fungal infection (Moniliophthora perniciosa).",
        "treatment": "Pruning infected parts, resistant varieties, fungicides. Strict phytosanitary measures are crucial.",
        "severity": "High - can significantly reduce yield and lead to tree death if untreated."
    },
    "Black_Pod_Rot": {
        "description": "Black Pod Rot, caused by various Phytophthora species, is characterized by rapidly spreading black or brown lesions on cacao pods, eventually covering the entire pod. It also affects flowers, cushions, and leaves.",
        "causes": "Fungal-like oomycete infection (Phytophthora spp.), favored by high humidity and rainfall.",
        "treatment": "Sanitation (removing infected pods), copper-based fungicides, good drainage, wider spacing of trees. Breeding for resistance is ongoing.",
        "severity": "High - major cause of yield loss globally."
    },
    "Frosty_Pod_Rot": {
        "description": "Frosty Pod Rot, caused by the fungus Moniliophthora roreri, initially appears as small, water-soaked spots on pods, which then develop a white, powdery fungal growth (like frost). The internal beans turn black and rot, making the pods commercially worthless.",
        "causes": "Fungal infection (Moniliophthora roreri), spread by wind and rain.",
        "treatment": "Strict sanitation (removing mummified pods), shade management, pruning, and developing resistant varieties. Fungicides are generally less effective once symptoms appear.",
        "severity": "High - causes severe economic losses in affected regions."
    }
}


@st.cache_data
def load_images_from_web_simulated():
    """
    Simulates loading images from a web source for training purposes.
    
    ***IMPORTANT: For real deployment, you must replace the dummy image generation
    with actual code to fetch images from PUBLIC, DIRECT web URLs.
    Google Drive folder links are not directly accessible.***
    """
    
    X = []
    labels = []
    
    label_to_index = {name: i for i, name in enumerate(CLASS_NAMES)}
    
    for item in IMAGE_FILE_DATA:
        # --- PLACEHOLDER FOR REAL WEB FETCHING LOGIC ---
        # For simulation, we generate a dummy image:
        dummy_img = np.random.rand(IMAGE_SIZE[0], IMAGE_SIZE[1], 3).astype(np.float32) * 255.0
        X.append(dummy_img)
        labels.append(item['label'])

    X = np.array(X, dtype=np.float32) / 255.0 # Normalize dummy data
    label_indices = np.array([label_to_index[label] for label in labels])
    y = tf.keras.utils.to_categorical(label_indices, num_classes=NUM_CLASSES)
    
    return X, y

# --- 2. Model Definition and Training ---

@st.cache_resource
def build_and_train_model():
    """Builds, compiles, and trains the CNN model."""
    st.info("Simulating data loading from a web source and processing...")
    
    X, y = load_images_from_web_simulated()

    # Split the limited dataset, ensuring stratification for better class representation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=np.argmax(y, axis=1)
    )

    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', name='last_conv_layer'), # Target layer for Grad-CAM
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    st.info("Training CNN Model... (Using simulated web-sourced data)")
    
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        verbose=0
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    try:
        report = classification_report(y_true_classes, y_pred_classes, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    except ValueError:
        report = {'accuracy': 0.0, 'macro avg': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0}}
    
    conf_mat = confusion_matrix(y_true_classes, y_pred_classes)

    return model, history, acc, report, conf_mat, X_test, y_test

# --- 3. Advanced Grad-CAM Implementation ---

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Computes and returns the Grad-CAM heatmap.
    """
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array[np.newaxis, ...]) 
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Ensure heatmap is not all zeros before division
    if tf.reduce_max(heatmap) > 1e-8:
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    else:
        heatmap = tf.zeros_like(heatmap) # Return all zeros if no activation
        
    heatmap = heatmap.numpy()

    return heatmap, preds[0].numpy()

def display_gradcam(img, heatmap, alpha=0.5):
    """
    Overlays the heatmap on the original image and returns a new image (as bytes).
    """
    if 'cv2' not in globals():
        return None, "OpenCV (cv2) is not available to generate the Grad-CAM visualization."

    heatmap = np.uint8(255 * heatmap)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
    resized_heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    colormap = cv2.COLORMAP_JET
    heatmap_jet = cv2.applyColorMap(resized_heatmap, colormap)

    superimposed_img_bgr = cv2.addWeighted(img_bgr, 1.0 - alpha, heatmap_jet, alpha, 0)

    superimposed_img_rgb = cv2.cvtColor(superimposed_img_bgr, cv2.COLOR_BGR2RGB)

    is_success, buffer = cv2.imencode(".png", cv2.cvtColor(superimposed_img_rgb, cv2.COLOR_RGB2BGR))
    if is_success:
        return buffer.tobytes(), None
    return None, "Failed to encode Grad-CAM image."

def process_uploaded_image(uploaded_file, model):
    """Handles uploaded file, runs prediction, and generates Grad-CAM."""
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
    if 'cv2' not in globals():
        st.error("Cannot proceed: OpenCV (cv2) is required for image processing.")
        return "N/A", 0.0, None

    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # Ensure image is not None (decoding failed)
    if img_bgr is None:
        st.error("Uploaded file could not be decoded as an image. Please check the file format.")
        return "N/A", 0.0, None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img_model_input = cv2.resize(img_rgb, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    img_array = img_model_input.astype('float32') / 255.0

    LAST_CONV_LAYER_NAME = 'last_conv_layer'
    heatmap, preds = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER_NAME)

    predicted_class_index = np.argmax(preds)
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = preds[predicted_class_index] * 100

    gradcam_img_bytes, error = display_gradcam(img_rgb, heatmap, alpha=0.5)
    
    if error:
        st.error(error)
        return predicted_class, confidence, None

    return predicted_class, confidence, gradcam_img_bytes

# --- 4. Streamlit Application ---

def main():
    st.set_page_config(layout="wide", page_title="Advanced Cacao Disease Detector")

    st.title("🌿 Cacao Disease Detector & Explainable AI Dashboard")
    st.markdown("Upload an image of your cacao plant or pod to detect common diseases (or confirm health) and see *where* the model is looking with Grad-CAM.")
    st.info("The model is trained using a simulated dataset to demonstrate functionality. For real-world use, replace the dummy data loading with actual image fetching from a public web source.")
    st.warning("🚨 **IMPORTANT:** Direct Google Drive folder links cannot be used for data loading in deployed Streamlit apps. Please host your data publicly (e.g., on GitHub raw links or dedicated image hosting) and update the `load_images_from_web_simulated` function.")

    # --- Sidebar for Model Metrics ---
    with st.sidebar:
        st.title("Model Training & Metrics")
        st.info(f"Model trained on **{NUM_CLASSES}** classes: {', '.join(CLASS_NAMES)}")

        with st.spinner("Building and training the model..."):
            model, history, acc, report, conf_mat, X_test, y_test = build_and_train_model()
            st.success("Model trained successfully!")

        st.header("Model Validation Results (Simulated)")
        st.markdown(f"**Test Set Accuracy:** **{acc:.2f}**")
        
        if history.history:
            st.markdown(f"**Test Loss:** **{history.history['val_loss'][-1]:.4f}**")

        with st.expander("Detailed Classification Report"):
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)

        if SEABORN_AVAILABLE:
            with st.expander("Confusion Matrix"):
                fig, ax = plt.subplots(figsize=(6, 6))
                
                y_pred_classes_cmat = np.argmax(model.predict(X_test, verbose=0), axis=1)
                y_true_classes_cmat = np.argmax(y_test, axis=1)

                cmat_disp = confusion_matrix(y_true_classes_cmat, y_pred_classes_cmat)
                
                sns.heatmap(cmat_disp, annot=True, fmt='d', cmap='Blues', cbar=False,
                            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                ax.set_title('Confusion Matrix (Simulated)')
                st.pyplot(fig)
        else:
            st.warning("Seaborn is required to display the Confusion Matrix plot.")

    # --- Main App for Prediction ---
    st.header("Upload Cacao Image for Analysis")

    uploaded_file = st.file_uploader(
        "Upload an image of a cacao leaf or pod (JPEG/PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        st.subheader("Analysis Results")
        
        # Create columns for original image, Grad-CAM, and description
        col_img, col_gradcam, col_description = st.columns([1, 1, 1.5]) # Adjusted column width for description

        # Process and Predict
        predicted_class, confidence, gradcam_img_bytes = process_uploaded_image(uploaded_file, model)
        
        with col_img:
            st.markdown("<p style='text-align: center; font-size: 1.25rem; font-weight: bold;'>Your Uploaded Image</p>", unsafe_allow_html=True)
            st.image(uploaded_file, use_column_width=True)
            st.markdown(f"**Predicted Status:** <span style='color: #1E88E5; font-size: 1.5rem;'>**{predicted_class}**</span>", unsafe_allow_html=True)
            st.markdown(f"**Confidence:** <span style='color: #4CAF50; font-size: 1.5rem;'>**{confidence:.2f}%**</span>", unsafe_allow_html=True)

        with col_gradcam:
            st.markdown("<p style='text-align: center; font-size: 1.25rem; font-weight: bold;'>Model Focus (Grad-CAM)</p>", unsafe_allow_html=True)
            if gradcam_img_bytes:
                base64_img = base64.b64encode(gradcam_img_bytes).decode('utf-8')
                st.image(f"data:image/png;base64,{base64_img}", use_column_width=True)
                st.markdown("""
                <small>The **red/yellow areas** indicate where the model focused its attention to make the prediction. This acts like an **arrow pointing to the key visual evidence** for the detected disease or health.</small>
                """, unsafe_allow_html=True)
            else:
                st.error("Could not generate Grad-CAM visualization. Ensure OpenCV is installed.")
        
        with col_description:
            st.markdown("<p style='text-align: center; font-size: 1.25rem; font-weight: bold;'>Detailed Information</p>", unsafe_allow_html=True)
            if predicted_class in DISEASE_DESCRIPTIONS:
                disease_info = DISEASE_DESCRIPTIONS[predicted_class]
                st.markdown(f"### {predicted_class.replace('_', ' ')}")
                st.markdown(f"**Description:** {disease_info.get('description', 'No detailed description available.')}")
                if disease_info.get('causes'):
                    st.markdown(f"**Causes:** {disease_info['causes']}")
                if disease_info.get('treatment'):
                    st.markdown(f"**Treatment/Management:** {disease_info['treatment']}")
                if disease_info.get('severity'):
                    st.markdown(f"**Severity:** {disease_info['severity']}")
            else:
                st.info("No specific description available for this prediction. Please refer to general cacao plant care.")

    else:
        st.info("Please upload an image of a cacao plant or pod to get started!")

if __name__ == '__main__':
    main()    Simulates loading images from a web source for training purposes.
    
    ***IMPORTANT: For real deployment, you must replace the dummy image generation
    with actual code to fetch images from PUBLIC, DIRECT web URLs.
    Google Drive folder links are not directly accessible.***
    """
    
    X = []
    labels = []
    
    label_to_index = {name: i for i, name in enumerate(CLASS_NAMES)}
    
    for item in IMAGE_FILE_DATA:
        # --- PLACEHOLDER FOR REAL WEB FETCHING LOGIC ---
        # For simulation, we generate a dummy image:
        dummy_img = np.random.rand(IMAGE_SIZE[0], IMAGE_SIZE[1], 3).astype(np.float32) * 255.0
        X.append(dummy_img)
        labels.append(item['label'])

    X = np.array(X, dtype=np.float32) / 255.0 # Normalize dummy data
    label_indices = np.array([label_to_index[label] for label in labels])
    y = tf.keras.utils.to_categorical(label_indices, num_classes=NUM_CLASSES)
    
    return X, y

# --- 2. Model Definition and Training ---

@st.cache_resource
def build_and_train_model():
    """Builds, compiles, and trains the CNN model."""
    st.info("Simulating data loading from a web source and processing...")
    
    X, y = load_images_from_web_simulated()

    # Split the limited dataset, ensuring stratification for better class representation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=np.argmax(y, axis=1)
    )

    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', name='last_conv_layer'), # Target layer for Grad-CAM
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    st.info("Training CNN Model... (Using simulated web-sourced data)")
    
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        verbose=0
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    try:
        report = classification_report(y_true_classes, y_pred_classes, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    except ValueError:
        report = {'accuracy': 0.0, 'macro avg': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0}}
    
    conf_mat = confusion_matrix(y_true_classes, y_pred_classes)

    return model, history, acc, report, conf_mat, X_test, y_test

# --- 3. Advanced Grad-CAM Implementation ---

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Computes and returns the Grad-CAM heatmap.
    """
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array[np.newaxis, ...]) 
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Ensure heatmap is not all zeros before division
    if tf.reduce_max(heatmap) > 1e-8:
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    else:
        heatmap = tf.zeros_like(heatmap) # Return all zeros if no activation
        
    heatmap = heatmap.numpy()

    return heatmap, preds[0].numpy()

def display_gradcam(img, heatmap, alpha=0.5):
    """
    Overlays the heatmap on the original image and returns a new image (as bytes).
    """
    if 'cv2' not in globals():
        return None, "OpenCV (cv2) is not available to generate the Grad-CAM visualization."

    heatmap = np.uint8(255 * heatmap)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
    resized_heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    colormap = cv2.COLORMAP_JET
    heatmap_jet = cv2.applyColorMap(resized_heatmap, colormap)

    superimposed_img_bgr = cv2.addWeighted(img_bgr, 1.0 - alpha, heatmap_jet, alpha, 0)

    superimposed_img_rgb = cv2.cvtColor(superimposed_img_bgr, cv2.COLOR_BGR2RGB)

    is_success, buffer = cv2.imencode(".png", cv2.cvtColor(superimposed_img_rgb, cv2.COLOR_RGB2BGR))
    if is_success:
        return buffer.tobytes(), None
    return None, "Failed to encode Grad-CAM image."

def process_uploaded_image(uploaded_file, model):
    """Handles uploaded file, runs prediction, and generates Grad-CAM."""
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
    if 'cv2' not in globals():
        st.error("Cannot proceed: OpenCV (cv2) is required for image processing.")
        return "N/A", 0.0, None

    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # Ensure image is not None (decoding failed)
    if img_bgr is None:
        st.error("Uploaded file could not be decoded as an image. Please check the file format.")
        return "N/A", 0.0, None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img_model_input = cv2.resize(img_rgb, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    img_array = img_model_input.astype('float32') / 255.0

    LAST_CONV_LAYER_NAME = 'last_conv_layer'
    heatmap, preds = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER_NAME)

    predicted_class_index = np.argmax(preds)
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = preds[predicted_class_index] * 100

    gradcam_img_bytes, error = display_gradcam(img_rgb, heatmap, alpha=0.5)
    
    if error:
        st.error(error)
        return predicted_class, confidence, None

    return predicted_class, confidence, gradcam_img_bytes

# --- 4. Streamlit Application ---

def main():
    st.set_page_config(layout="wide", page_title="Advanced Cacao Disease Detector")

    st.title("🌿 Cacao Disease Detector & Explainable AI Dashboard")
    st.markdown("Upload an image of your cacao plant or pod to detect common diseases (or confirm health) and see *where* the model is looking with Grad-CAM.")
    st.info("The model is trained using a simulated dataset to demonstrate functionality. For real-world use, replace the dummy data loading with actual image fetching from a public web source.")
    st.warning("🚨 **IMPORTANT:** Direct Google Drive folder links cannot be used for data loading in deployed Streamlit apps. Please host your data publicly (e.g., on GitHub raw links or dedicated image hosting) and update the `load_images_from_web_simulated` function.")

    # --- Sidebar for Model Metrics ---
    with st.sidebar:
        st.title("Model Training & Metrics")
        st.info(f"Model trained on **{NUM_CLASSES}** classes: {', '.join(CLASS_NAMES)}")

        with st.spinner("Building and training the model..."):
            model, history, acc, report, conf_mat, X_test, y_test = build_and_train_model()
            st.success("Model trained successfully!")

        st.header("Model Validation Results (Simulated)")
        st.markdown(f"**Test Set Accuracy:** **{acc:.2f}**")
        
        if history.history:
            st.markdown(f"**Test Loss:** **{history.history['val_loss'][-1]:.4f}**")

        with st.expander("Detailed Classification Report"):
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)

        if SEABORN_AVAILABLE:
            with st.expander("Confusion Matrix"):
                fig, ax = plt.subplots(figsize=(6, 6))
                
                y_pred_classes_cmat = np.argmax(model.predict(X_test, verbose=0), axis=1)
                y_true_classes_cmat = np.argmax(y_test, axis=1)

                cmat_disp = confusion_matrix(y_true_classes_cmat, y_pred_classes_cmat)
                
                sns.heatmap(cmat_disp, annot=True, fmt='d', cmap='Blues', cbar=False,
                            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                ax.set_title('Confusion Matrix (Simulated)')
                st.pyplot(fig)
        else:
            st.warning("Seaborn is required to display the Confusion Matrix plot.")

    # --- Main App for Prediction ---
    st.header("Upload Cacao Image for Analysis")

    uploaded_file = st.file_uploader(
        "Upload an image of a cacao leaf or pod (JPEG/PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        st.subheader("Analysis Results")
        
        # Create columns for original image, Grad-CAM, and description
        col_img, col_gradcam, col_description = st.columns([1, 1, 1.5]) # Adjusted column width for description

        # Process and Predict
        predicted_class, confidence, gradcam_img_bytes = process_uploaded_image(uploaded_file, model)
        
        with col_img:
            st.markdown("<p style='text-align: center; font-size: 1.25rem; font-weight: bold;'>Your Uploaded Image</p>", unsafe_allow_html=True)
            st.image(uploaded_file, use_column_width=True)
            st.markdown(f"**Predicted Status:** <span style='color: #1E88E5; font-size: 1.5rem;'>**{predicted_class}**</span>", unsafe_allow_html=True)
            st.markdown(f"**Confidence:** <span style='color: #4CAF50; font-size: 1.5rem;'>**{confidence:.2f}%**</span>", unsafe_allow_html=True)

        with col_gradcam:
            st.markdown("<p style='text-align: center; font-size: 1.25rem; font-weight: bold;'>Model Focus (Grad-CAM)</p>", unsafe_allow_html=True)
            if gradcam_img_bytes:
                base64_img = base64.b64encode(gradcam_img_bytes).decode('utf-8')
                st.image(f"data:image/png;base64,{base64_img}", use_column_width=True)
                st.markdown("""
                <small>The **red/yellow areas** indicate where the model focused its attention to make the prediction. This acts like an **arrow pointing to the key visual evidence** for the detected disease or health.</small>
                """, unsafe_allow_html=True)
            else:
                st.error("Could not generate Grad-CAM visualization. Ensure OpenCV is installed.")
        
        with col_description:
            st.markdown("<p style='text-align: center; font-size: 1.25rem; font-weight: bold;'>Detailed Information</p>", unsafe_allow_html=True)
            if predicted_class in DISEASE_DESCRIPTIONS:
                disease_info = DISEASE_DESCRIPTIONS[predicted_class]
                st.markdown(f"### {predicted_class.replace('_', ' ')}")
                st.markdown(f"**Description:** {disease_info.get('description', 'No detailed description available.')}")
                if disease_info.get('causes'):
                    st.markdown(f"**Causes:** {disease_info['causes']}")
                if disease_info.get('treatment'):
                    st.markdown(f"**Treatment/Management:** {disease_info['treatment']}")
                if disease_info.get('severity'):
                    st.markdown(f"**Severity:** {disease_info['severity']}")
            else:
                st.info("No specific description available for this prediction. Please refer to general cacao plant care.")

    else:
        st.info("Please upload an image of a cacao plant or pod to get started!")

if __name__ == '__main__':
    main()    {"filename": "frosty pod rot_00003.png", "label": "Frosty_Pod_Rot"},
    {"filename": "healthy_00003.png", "label": "Healthy"},
]

CLASS_NAMES = sorted(list(set(item['label'] for item in IMAGE_FILE_DATA)))
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 4 # Reduced batch size due to very small dataset
EPOCHS = 15 # Increased epochs slightly for potentially better (simulated) learning


# Placeholder for disease descriptions
DISEASE_DESCRIPTIONS = {
    "Healthy": {
        "description": "The cacao plant shows no signs of disease and appears to be thriving. Healthy leaves are typically green, vibrant, and free from spots or discoloration.",
        "causes": "Proper nutrition, adequate water, good sunlight, and protection from pests contribute to a healthy plant.",
        "treatment": "Maintain optimal growing conditions."
    },
    "Witches_Broom": {
        "description": "Witches' Broom disease, caused by the fungus Moniliophthora perniciosa, leads to abnormal, dense growths of shoots or 'brooms' and swollen stems. It can also affect pods, causing premature ripening and malformation.",
        "causes": "Fungal infection (Moniliophthora perniciosa).",
        "treatment": "Pruning infected parts, resistant varieties, fungicides. Strict phytosanitary measures are crucial.",
        "severity": "High - can significantly reduce yield and lead to tree death if untreated."
    },
    "Black_Pod_Rot": {
        "description": "Black Pod Rot, caused by various Phytophthora species, is characterized by rapidly spreading black or brown lesions on cacao pods, eventually covering the entire pod. It also affects flowers, cushions, and leaves.",
        "causes": "Fungal-like oomycete infection (Phytophthora spp.), favored by high humidity and rainfall.",
        "treatment": "Sanitation (removing infected pods), copper-based fungicides, good drainage, wider spacing of trees. Breeding for resistance is ongoing.",
        "severity": "High - major cause of yield loss globally."
    },
    "Frosty_Pod_Rot": {
        "description": "Frosty Pod Rot, caused by the fungus Moniliophthora roreri, initially appears as small, water-soaked spots on pods, which then develop a white, powdery fungal growth (like frost). The internal beans turn black and rot, making the pods commercially worthless.",
        "causes": "Fungal infection (Moniliophthora roreri), spread by wind and rain.",
        "treatment": "Strict sanitation (removing mummified pods), shade management, pruning, and developing resistant varieties. Fungicides are generally less effective once symptoms appear.",
        "severity": "High - causes severe economic losses in affected regions."
    }
}


@st.cache_data
def load_images_from_web_simulated():
    """
    Simulates loading images from a web source for training purposes.
    
    ***IMPORTANT: For real deployment, you must replace the dummy image generation
    with actual code to fetch images from PUBLIC, DIRECT web URLs.
    Google Drive folder links are not directly accessible.***
    """
    
    X = []
    labels = []
    
    label_to_index = {name: i for i, name in enumerate(CLASS_NAMES)}
    
    for item in IMAGE_FILE_DATA:
        # --- PLACEHOLDER FOR REAL WEB FETCHING LOGIC ---
        # For simulation, we generate a dummy image:
        dummy_img = np.random.rand(IMAGE_SIZE[0], IMAGE_SIZE[1], 3).astype(np.float32) * 255.0
        X.append(dummy_img)
        labels.append(item['label'])

    X = np.array(X, dtype=np.float32) / 255.0 # Normalize dummy data
    label_indices = np.array([label_to_index[label] for label in labels])
    y = tf.keras.utils.to_categorical(label_indices, num_classes=NUM_CLASSES)
    
    return X, y

# --- 2. Model Definition and Training ---

@st.cache_resource
def build_and_train_model():
    """Builds, compiles, and trains the CNN model."""
    st.info("Simulating data loading from a web source and processing...")
    
    X, y = load_images_from_web_simulated()

    # Split the limited dataset, ensuring stratification for better class representation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=np.argmax(y, axis=1)
    )

    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', name='last_conv_layer'), # Target layer for Grad-CAM
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    st.info("Training CNN Model... (Using simulated web-sourced data)")
    
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        verbose=0
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    try:
        report = classification_report(y_true_classes, y_pred_classes, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    except ValueError:
        report = {'accuracy': 0.0, 'macro avg': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0}}
    
    conf_mat = confusion_matrix(y_true_classes, y_pred_classes)

    return model, history, acc, report, conf_mat, X_test, y_test

# --- 3. Advanced Grad-CAM Implementation ---

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Computes and returns the Grad-CAM heatmap.
    """
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array[np.newaxis, ...]) 
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Ensure heatmap is not all zeros before division
    if tf.reduce_max(heatmap) > 1e-8:
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    else:
        heatmap = tf.zeros_like(heatmap) # Return all zeros if no activation
        
    heatmap = heatmap.numpy()

    return heatmap, preds[0].numpy()

def display_gradcam(img, heatmap, alpha=0.5):
    """
    Overlays the heatmap on the original image and returns a new image (as bytes).
    """
    if 'cv2' not in globals():
        return None, "OpenCV (cv2) is not available to generate the Grad-CAM visualization."

    heatmap = np.uint8(255 * heatmap)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
    resized_heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    colormap = cv2.COLORMAP_JET
    heatmap_jet = cv2.applyColorMap(resized_heatmap, colormap)

    superimposed_img_bgr = cv2.addWeighted(img_bgr, 1.0 - alpha, heatmap_jet, alpha, 0)

    superimposed_img_rgb = cv2.cvtColor(superimposed_img_bgr, cv2.COLOR_BGR2RGB)

    is_success, buffer = cv2.imencode(".png", cv2.cvtColor(superimposed_img_rgb, cv2.COLOR_RGB2BGR))
    if is_success:
        return buffer.tobytes(), None
    return None, "Failed to encode Grad-CAM image."

def process_uploaded_image(uploaded_file, model):
    """Handles uploaded file, runs prediction, and generates Grad-CAM."""
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
    if 'cv2' not in globals():
        st.error("Cannot proceed: OpenCV (cv2) is required for image processing.")
        return "N/A", 0.0, None

    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # Ensure image is not None (decoding failed)
    if img_bgr is None:
        st.error("Uploaded file could not be decoded as an image. Please check the file format.")
        return "N/A", 0.0, None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img_model_input = cv2.resize(img_rgb, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    img_array = img_model_input.astype('float32') / 255.0

    LAST_CONV_LAYER_NAME = 'last_conv_layer'
    heatmap, preds = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER_NAME)

    predicted_class_index = np.argmax(preds)
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = preds[predicted_class_index] * 100

    gradcam_img_bytes, error = display_gradcam(img_rgb, heatmap, alpha=0.5)
    
    if error:
        st.error(error)
        return predicted_class, confidence, None

    return predicted_class, confidence, gradcam_img_bytes

# --- 4. Streamlit Application ---

def main():
    st.set_page_config(layout="wide", page_title="Advanced Cacao Disease Detector")

    st.title("🌿 Cacao Disease Detector & Explainable AI Dashboard")
    st.markdown("Upload an image of your cacao plant or pod to detect common diseases (or confirm health) and see *where* the model is looking with Grad-CAM.")
    st.info("The model is trained using a simulated dataset to demonstrate functionality. For real-world use, replace the dummy data loading with actual image fetching from a public web source.")
    st.warning("🚨 **IMPORTANT:** Direct Google Drive folder links cannot be used for data loading in deployed Streamlit apps. Please host your data publicly (e.g., on GitHub raw links or dedicated image hosting) and update the `load_images_from_web_simulated` function.")

    # --- Sidebar for Model Metrics ---
    with st.sidebar:
        st.title("Model Training & Metrics")
        st.info(f"Model trained on **{NUM_CLASSES}** classes: {', '.join(CLASS_NAMES)}")

        with st.spinner("Building and training the model..."):
            model, history, acc, report, conf_mat, X_test, y_test = build_and_train_model()
            st.success("Model trained successfully!")

        st.header("Model Validation Results (Simulated)")
        st.markdown(f"**Test Set Accuracy:** **{acc:.2f}**")
        
        if history.history:
            st.markdown(f"**Test Loss:** **{history.history['val_loss'][-1]:.4f}**")

        with st.expander("Detailed Classification Report"):
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)

        if SEABORN_AVAILABLE:
            with st.expander("Confusion Matrix"):
                fig, ax = plt.subplots(figsize=(6, 6))
                
                y_pred_classes_cmat = np.argmax(model.predict(X_test, verbose=0), axis=1)
                y_true_classes_cmat = np.argmax(y_test, axis=1)

                cmat_disp = confusion_matrix(y_true_classes_cmat, y_pred_classes_cmat)
                
                sns.heatmap(cmat_disp, annot=True, fmt='d', cmap='Blues', cbar=False,
                            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                ax.set_title('Confusion Matrix (Simulated)')
                st.pyplot(fig)
        else:
            st.warning("Seaborn is required to display the Confusion Matrix plot.")

    # --- Main App for Prediction ---
    st.header("Upload Cacao Image for Analysis")

    uploaded_file = st.file_uploader(
        "Upload an image of a cacao leaf or pod (JPEG/PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        st.subheader("Analysis Results")
        
        # Create columns for original image, Grad-CAM, and description
        col_img, col_gradcam, col_description = st.columns([1, 1, 1.5]) # Adjusted column width for description

        # Process and Predict
        predicted_class, confidence, gradcam_img_bytes = process_uploaded_image(uploaded_file, model)
        
        with col_img:
            st.markdown("<p style='text-align: center; font-size: 1.25rem; font-weight: bold;'>Your Uploaded Image</p>", unsafe_allow_html=True)
            st.image(uploaded_file, use_column_width=True)
            st.markdown(f"**Predicted Status:** <span style='color: #1E88E5; font-size: 1.5rem;'>**{predicted_class}**</span>", unsafe_allow_html=True)
            st.markdown(f"**Confidence:** <span style='color: #4CAF50; font-size: 1.5rem;'>**{confidence:.2f}%**</span>", unsafe_allow_html=True)

        with col_gradcam:
            st.markdown("<p style='text-align: center; font-size: 1.25rem; font-weight: bold;'>Model Focus (Grad-CAM)</p>", unsafe_allow_html=True)
            if gradcam_img_bytes:
                base64_img = base64.b64encode(gradcam_img_bytes).decode('utf-8')
                st.image(f"data:image/png;base64,{base64_img}", use_column_width=True)
                st.markdown("""
                <small>The **red/yellow areas** indicate where the model focused its attention to make the prediction. This acts like an **arrow pointing to the key visual evidence** for the detected disease or health.</small>
                """, unsafe_allow_html=True)
            else:
                st.error("Could not generate Grad-CAM visualization. Ensure OpenCV is installed.")
        
        with col_description:
            st.markdown("<p style='text-align: center; font-size: 1.25rem; font-weight: bold;'>Detailed Information</p>", unsafe_allow_html=True)
            if predicted_class in DISEASE_DESCRIPTIONS:
                disease_info = DISEASE_DESCRIPTIONS[predicted_class]
                st.markdown(f"### {predicted_class.replace('_', ' ')}")
                st.markdown(f"**Description:** {disease_info.get('description', 'No detailed description available.')}")
                if disease_info.get('causes'):
                    st.markdown(f"**Causes:** {disease_info['causes']}")
                if disease_info.get('treatment'):
                    st.markdown(f"**Treatment/Management:** {disease_info['treatment']}")
                if disease_info.get('severity'):
                    st.markdown(f"**Severity:** {disease_info['severity']}")
            else:
                st.info("No specific description available for this prediction. Please refer to general cacao plant care.")

    else:
        st.info("Please upload an image of a cacao plant or pod to get started!")

if __name__ == '__main__':
    main()    {"filename": "frosty pod rot_00003.png", "label": "Frosty_Pod_Rot"},
    {"filename": "healthy_00003.png", "label": "Healthy"},
]

CLASS_NAMES = sorted(list(set(item['label'] for item in IMAGE_FILE_DATA)))
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 4 # Reduced batch size due to very small dataset
EPOCHS = 15 # Increased epochs slightly for potentially better (simulated) learning


# Placeholder for disease descriptions
DISEASE_DESCRIPTIONS = {
    "Healthy": {
        "description": "The cacao plant shows no signs of disease and appears to be thriving. Healthy leaves are typically green, vibrant, and free from spots or discoloration.",
        "causes": "Proper nutrition, adequate water, good sunlight, and protection from pests contribute to a healthy plant.",
        "treatment": "Maintain optimal growing conditions."
    },
    "Witches_Broom": {
        "description": "Witches' Broom disease, caused by the fungus Moniliophthora perniciosa, leads to abnormal, dense growths of shoots or 'brooms' and swollen stems. It can also affect pods, causing premature ripening and malformation.",
        "causes": "Fungal infection (Moniliophthora perniciosa).",
        "treatment": "Pruning infected parts, resistant varieties, fungicides. Strict phytosanitary measures are crucial.",
        "severity": "High - can significantly reduce yield and lead to tree death if untreated."
    },
    "Black_Pod_Rot": {
        "description": "Black Pod Rot, caused by various Phytophthora species, is characterized by rapidly spreading black or brown lesions on cacao pods, eventually covering the entire pod. It also affects flowers, cushions, and leaves.",
        "causes": "Fungal-like oomycete infection (Phytophthora spp.), favored by high humidity and rainfall.",
        "treatment": "Sanitation (removing infected pods), copper-based fungicides, good drainage, wider spacing of trees. Breeding for resistance is ongoing.",
        "severity": "High - major cause of yield loss globally."
    },
    "Frosty_Pod_Rot": {
        "description": "Frosty Pod Rot, caused by the fungus Moniliophthora roreri, initially appears as small, water-soaked spots on pods, which then develop a white, powdery fungal growth (like frost). The internal beans turn black and rot, making the pods commercially worthless.",
        "causes": "Fungal infection (Moniliophthora roreri), spread by wind and rain.",
        "treatment": "Strict sanitation (removing mummified pods), shade management, pruning, and developing resistant varieties. Fungicides are generally less effective once symptoms appear.",
        "severity": "High - causes severe economic losses in affected regions."
    } # Removed the extra '}' that likely caused the error.
}


@st.cache_data
def load_images_from_web_simulated():
    """
    Simulates loading images from a web source for training purposes.
    
    ***IMPORTANT: For real deployment, you must replace the dummy image generation
    with actual code to fetch images from PUBLIC, DIRECT web URLs.
    Google Drive folder links are not directly accessible.***
    """
    
    X = []
    labels = []
    
    label_to_index = {name: i for i, name in enumerate(CLASS_NAMES)}
    
    for item in IMAGE_FILE_DATA:
        # --- PLACEHOLDER FOR REAL WEB FETCHING LOGIC ---
        # For simulation, we generate a dummy image:
        dummy_img = np.random.rand(IMAGE_SIZE[0], IMAGE_SIZE[1], 3).astype(np.float32) * 255.0
        X.append(dummy_img)
        labels.append(item['label'])

    X = np.array(X, dtype=np.float32) / 255.0 # Normalize dummy data
    label_indices = np.array([label_to_index[label] for label in labels])
    y = tf.keras.utils.to_categorical(label_indices, num_classes=NUM_CLASSES)
    
    return X, y

# --- 2. Model Definition and Training ---

@st.cache_resource
def build_and_train_model():
    """Builds, compiles, and trains the CNN model."""
    st.info("Simulating data loading from a web source and processing...")
    
    X, y = load_images_from_web_simulated()

    # Split the limited dataset, ensuring stratification for better class representation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=np.argmax(y, axis=1)
    )

    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', name='last_conv_layer'), # Target layer for Grad-CAM
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    st.info("Training CNN Model... (Using simulated web-sourced data)")
    
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        verbose=0
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    try:
        report = classification_report(y_true_classes, y_pred_classes, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    except ValueError:
        report = {'accuracy': 0.0, 'macro avg': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0}}
    
    conf_mat = confusion_matrix(y_true_classes, y_pred_classes)

    return model, history, acc, report, conf_mat, X_test, y_test

# --- 3. Advanced Grad-CAM Implementation ---

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Computes and returns the Grad-CAM heatmap.
    """
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array[np.newaxis, ...]) 
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Ensure heatmap is not all zeros before division
    if tf.reduce_max(heatmap) > 1e-8:
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    else:
        heatmap = tf.zeros_like(heatmap) # Return all zeros if no activation
        
    heatmap = heatmap.numpy()

    return heatmap, preds[0].numpy()

def display_gradcam(img, heatmap, alpha=0.5):
    """
    Overlays the heatmap on the original image and returns a new image (as bytes).
    """
    if 'cv2' not in globals():
        return None, "OpenCV (cv2) is not available to generate the Grad-CAM visualization."

    heatmap = np.uint8(255 * heatmap)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
    resized_heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    colormap = cv2.COLORMAP_JET
    heatmap_jet = cv2.applyColorMap(resized_heatmap, colormap)

    superimposed_img_bgr = cv2.addWeighted(img_bgr, 1.0 - alpha, heatmap_jet, alpha, 0)

    superimposed_img_rgb = cv2.cvtColor(superimposed_img_bgr, cv2.COLOR_BGR2RGB)

    is_success, buffer = cv2.imencode(".png", cv2.cvtColor(superimposed_img_rgb, cv2.COLOR_RGB2BGR))
    if is_success:
        return buffer.tobytes(), None
    return None, "Failed to encode Grad-CAM image."

def process_uploaded_image(uploaded_file, model):
    """Handles uploaded file, runs prediction, and generates Grad-CAM."""
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
    if 'cv2' not in globals():
        st.error("Cannot proceed: OpenCV (cv2) is required for image processing.")
        return "N/A", 0.0, None

    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # Ensure image is not None (decoding failed)
    if img_bgr is None:
        st.error("Uploaded file could not be decoded as an image. Please check the file format.")
        return "N/A", 0.0, None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img_model_input = cv2.resize(img_rgb, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    img_array = img_model_input.astype('float32') / 255.0

    LAST_CONV_LAYER_NAME = 'last_conv_layer'
    heatmap, preds = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER_NAME)

    predicted_class_index = np.argmax(preds)
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = preds[predicted_class_index] * 100

    gradcam_img_bytes, error = display_gradcam(img_rgb, heatmap, alpha=0.5)
    
    if error:
        st.error(error)
        return predicted_class, confidence, None

    return predicted_class, confidence, gradcam_img_bytes

# --- 4. Streamlit Application ---

def main():
    st.set_page_config(layout="wide", page_title="Advanced Cacao Disease Detector")

    st.title("🌿 Cacao Disease Detector & Explainable AI Dashboard")
    st.markdown("Upload an image of your cacao plant or pod to detect common diseases (or confirm health) and see *where* the model is looking with Grad-CAM.")
    st.info("The model is trained using a simulated dataset to demonstrate functionality. For real-world use, replace the dummy data loading with actual image fetching from a public web source.")
    st.warning("🚨 **IMPORTANT:** Direct Google Drive folder links cannot be used for data loading in deployed Streamlit apps. Please host your data publicly (e.g., on GitHub raw links or dedicated image hosting) and update the `load_images_from_web_simulated` function.")

    # --- Sidebar for Model Metrics ---
    with st.sidebar:
        st.title("Model Training & Metrics")
        st.info(f"Model trained on **{NUM_CLASSES}** classes: {', '.join(CLASS_NAMES)}")

        with st.spinner("Building and training the model..."):
            model, history, acc, report, conf_mat, X_test, y_test = build_and_train_model()
            st.success("Model trained successfully!")

        st.header("Model Validation Results (Simulated)")
        st.markdown(f"**Test Set Accuracy:** **{acc:.2f}**")
        
        if history.history:
            st.markdown(f"**Test Loss:** **{history.history['val_loss'][-1]:.4f}**")

        with st.expander("Detailed Classification Report"):
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)

        if SEABORN_AVAILABLE:
            with st.expander("Confusion Matrix"):
                fig, ax = plt.subplots(figsize=(6, 6))
                
                y_pred_classes_cmat = np.argmax(model.predict(X_test, verbose=0), axis=1)
                y_true_classes_cmat = np.argmax(y_test, axis=1)

                cmat_disp = confusion_matrix(y_true_classes_cmat, y_pred_classes_cmat)
                
                sns.heatmap(cmat_disp, annot=True, fmt='d', cmap='Blues', cbar=False,
                            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                ax.set_title('Confusion Matrix (Simulated)')
                st.pyplot(fig)
        else:
            st.warning("Seaborn is required to display the Confusion Matrix plot.")

    # --- Main App for Prediction ---
    st.header("Upload Cacao Image for Analysis")

    uploaded_file = st.file_uploader(
        "Upload an image of a cacao leaf or pod (JPEG/PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        st.subheader("Analysis Results")
        
        # Create columns for original image, Grad-CAM, and description
        col_img, col_gradcam, col_description = st.columns([1, 1, 1.5]) # Adjusted column width for description

        # Process and Predict
        predicted_class, confidence, gradcam_img_bytes = process_uploaded_image(uploaded_file, model)
        
        with col_img:
            st.markdown("<p style='text-align: center; font-size: 1.25rem; font-weight: bold;'>Your Uploaded Image</p>", unsafe_allow_html=True)
            st.image(uploaded_file, use_column_width=True)
            st.markdown(f"**Predicted Status:** <span style='color: #1E88E5; font-size: 1.5rem;'>**{predicted_class}**</span>", unsafe_allow_html=True)
            st.markdown(f"**Confidence:** <span style='color: #4CAF50; font-size: 1.5rem;'>**{confidence:.2f}%**</span>", unsafe_allow_html=True)

        with col_gradcam:
            st.markdown("<p style='text-align: center; font-size: 1.25rem; font-weight: bold;'>Model Focus (Grad-CAM)</p>", unsafe_allow_html=True)
            if gradcam_img_bytes:
                base64_img = base64.b64encode(gradcam_img_bytes).decode('utf-8')
                st.image(f"data:image/png;base64,{base64_img}", use_column_width=True)
                st.markdown("""
                <small>The **red/yellow areas** indicate where the model focused its attention to make the prediction. This acts like an **arrow pointing to the key visual evidence** for the detected disease or health.</small>
                """, unsafe_allow_html=True)
            else:
                st.error("Could not generate Grad-CAM visualization. Ensure OpenCV is installed.")
        
        with col_description:
            st.markdown("<p style='text-align: center; font-size: 1.25rem; font-weight: bold;'>Detailed Information</p>", unsafe_allow_html=True)
            if predicted_class in DISEASE_DESCRIPTIONS:
                disease_info = DISEASE_DESCRIPTIONS[predicted_class]
                st.markdown(f"### {predicted_class.replace('_', ' ')}")
                st.markdown(f"**Description:** {disease_info.get('description', 'No detailed description available.')}")
                if disease_info.get('causes'):
                    st.markdown(f"**Causes:** {disease_info['causes']}")
                if disease_info.get('treatment'):
                    st.markdown(f"**Treatment/Management:** {disease_info['treatment']}")
                if disease_info.get('severity'):
                    st.markdown(f"**Severity:** {disease_info['severity']}")
            else:
                st.info("No specific description available for this prediction. Please refer to general cacao plant care.")

    else:
        st.info("Please upload an image of a cacao plant or pod to get started!")

if __name__ == '__main__':
    main()    "black pod rot_00003.png": "Black_Pod_Rot",
    "frosty pod rot_00002.png": "Frosty_Pod_Rot",
    "frosty pod rot_00003.png": "Frosty_Pod_Rot",
    "healthy_00003.png": "Healthy",


CLASS_NAMES = sorted(list(set(IMAGE_CLASS_DEFINITIONS.values())))
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 4
EPOCHS = 10 

@st.cache_data
def generate_simulated_training_data():
    """
    Generates dummy data for training simulation.
    This simulates having actual images from various classes.
    """
    num_samples_per_class = 5 # Generate more samples per class for better simulation
    total_samples = num_samples_per_class * NUM_CLASSES
    
    X = []
    y_labels = []

    for class_name in CLASS_NAMES:
        for _ in range(num_samples_per_class):
            dummy_img = np.random.rand(IMAGE_SIZE[0], IMAGE_SIZE[1], 3).astype(np.float32) * 255.0
            X.append(dummy_img)
            y_labels.append(class_name)

    X = np.array(X, dtype=np.float32) / 255.0 # Normalize dummy data
    
    label_to_index = {name: i for i, name in enumerate(CLASS_NAMES)}
    label_indices = np.array([label_to_index[label] for label in y_labels])

    y = tf.keras.utils.to_categorical(label_indices, num_classes=NUM_CLASSES)
    
    return X, y

# --- 2. Model Definition and Training ---

@st.cache_resource
def build_and_train_model():
    """Builds, compiles, and trains the CNN model."""
    st.info("Generating and processing simulated training data...")
    
    X, y = generate_simulated_training_data()

    # Split the limited dataset
    # Ensure there's enough data for splitting, especially with small `num_samples_per_class`
    if len(X) < 2 * NUM_CLASSES: # At least two samples per class for stratify to work
        st.warning("Very few samples for training, model performance will be highly unreliable. Using a simpler split.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=np.argmax(y, axis=1))


    # Define the CNN Model
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', name='last_conv_layer'), # Target layer for Grad-CAM
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    st.info("Training CNN Model with simulated data... (Accuracy will be random)")
    
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        verbose=0
    )

    # Evaluate the model
    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    # Generate Classification Report and Confusion Matrix
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    try:
        report = classification_report(y_true_classes, y_pred_classes, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    except ValueError: # Fallback if classification report can't be generated (e.g., single class in split)
        report = {'accuracy': 0.0, 'macro avg': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0}}
    
    conf_mat = confusion_matrix(y_true_classes, y_pred_classes)

    return model, history, acc, report, conf_mat, X_test, y_test

# --- 3. Advanced Grad-CAM Implementation ---

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Computes and returns the Grad-CAM heatmap.
    """
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array[np.newaxis, ...])
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    return heatmap, preds[0].numpy()

def display_gradcam(img, heatmap, alpha=0.5):
    """
    Overlays the heatmap on the original image and returns a new image (as bytes).
    """
    if 'cv2' not in globals():
        return None, "OpenCV (cv2) is not available to generate the Grad-CAM visualization."

    heatmap = np.uint8(255 * heatmap)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
    resized_heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    colormap = cv2.COLORMAP_JET
    heatmap_jet = cv2.applyColorMap(resized_heatmap, colormap)

    superimposed_img_bgr = cv2.addWeighted(img_bgr, 1.0 - alpha, heatmap_jet, alpha, 0)
    superimposed_img_rgb = cv2.cvtColor(superimposed_img_bgr, cv2.COLOR_BGR2RGB)

    is_success, buffer = cv2.imencode(".png", cv2.cvtColor(superimposed_img_rgb, cv2.COLOR_RGB2BGR))
    if is_success:
        return buffer.tobytes(), None
    return None, "Failed to encode Grad-CAM image."

def process_uploaded_image(uploaded_file, model):
    """Handles uploaded file, runs prediction, and generates Grad-CAM."""
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
    if 'cv2' not in globals():
        st.error("Cannot proceed: OpenCV (cv2) is required for image processing but could not be imported. Please ensure 'opencv-python' is installed.")
        return "N/A", 0.0, None

    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img_model_input = cv2.resize(img_rgb, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    img_array = img_model_input.astype('float32') / 255.0

    LAST_CONV_LAYER_NAME = 'last_conv_layer'
    heatmap, preds = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER_NAME)

    predicted_class_index = np.argmax(preds)
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = preds[predicted_class_index] * 100

    gradcam_img_bytes, error = display_gradcam(img_rgb, heatmap, alpha=0.5)
    
    if error:
        st.error(error)
        return predicted_class, confidence, None

    return predicted_class, confidence, gradcam_img_bytes

# --- 4. Streamlit Application ---

def main():
    st.set_page_config(layout="wide", page_title="Cacao Disease Detector")

    st.title("🌿 Cacao Disease Detector & Grad-CAM Visualization")
    st.markdown("This CNN model detects Cacao plant diseases. **Upload an image from your device** below to see its prediction and a visual explanation (Grad-CAM).")
    st.info("Note: The model is trained on **simulated data** for demonstration purposes. Real-world performance requires training on a large, diverse dataset of actual Cacao images.")

    # --- Sidebar for Model Metrics ---
    st.sidebar.title("Model Training & Metrics (Simulated)")
    st.sidebar.info(f"Model trained on **{NUM_CLASSES}** classes: {', '.join(CLASS_NAMES)}")

    with st.spinner("Building and training the model with simulated data..."):
        model, history, acc, report, conf_mat, X_test, y_test = build_and_train_model()
        st.success("Simulated model trained successfully!")

    st.sidebar.header("Model Validation Results (Simulated)")
    st.sidebar.markdown(f"**Test Set Accuracy:** **{acc:.2f}**")
    
    if history.history:
        st.sidebar.markdown(f"**Test Loss:** **{history.history['val_loss'][-1]:.4f}**")

    with st.sidebar.expander("Detailed Classification Report"):
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)

    if SEABORN_AVAILABLE:
        with st.sidebar.expander("Confusion Matrix"):
            fig, ax = plt.subplots(figsize=(6, 6))
            
            y_pred_classes_cmat = np.argmax(model.predict(X_test, verbose=0), axis=1)
            y_true_classes_cmat = np.argmax(y_test, axis=1)

            cmat_disp = confusion_matrix(y_true_classes_cmat, y_pred_classes_cmat)
            
            sns.heatmap(cmat_disp, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title('Confusion Matrix (Simulated)')
            st.pyplot(fig)
    else:
         st.sidebar.warning("Seaborn is required to display the Confusion Matrix plot.")

    # --- Main App for Prediction ---
    st.header("⬆️ Upload Your Cacao Image Here")
    st.markdown("Upload a photo of your Cacao plant or pod. The model will analyze it for diseases or classify it as healthy.")

    uploaded_file = st.file_uploader(
        "Choose an image file (JPEG/PNG) from your device",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        st.subheader("Analysis Results:")
        col1, col2 = st.columns(2)

        predicted_class, confidence, gradcam_img_bytes = process_uploaded_image(uploaded_file, model)
        
        with col1:
            st.markdown("<p style='text-align: center; font-size: 1.25rem; font-weight: bold;'>Your Uploaded Image</p>", unsafe_allow_html=True)
            st.image(uploaded_file, use_column_width=True)
            st.markdown(f"**Predicted Status:** <span style='color: #1E88E5; font-size: 1.5rem;'>**{predicted_class}**</span>", unsafe_allow_html=True)
            st.markdown(f"**Confidence:** <span style='color: #4CAF50; font-size: 1.5rem;'>**{confidence:.2f}%**</span>", unsafe_allow_html=True)


        with col2:
            st.markdown("<p style='text-align: center; font-size: 1.25rem; font-weight: bold;'>Grad-CAM Explanation</p>", unsafe_allow_html=True)
            if gradcam_img_bytes:
                base64_img = base64.b64encode(gradcam_img_bytes).decode('utf-8')
                st.image(f"data:image/png;base64,{base64_img}", use_column_width=True)
            else:
                st.error("Could not generate Grad-CAM visualization. Ensure OpenCV is installed.")

            st.markdown("""
            **What is Grad-CAM?**
            The Gradient-weighted Class Activation Map (Grad-CAM) highlights the areas in your image (red/yellow) that the model found most important when making its prediction. This helps you understand *why* the model made its decision.
            """, unsafe_allow_html=True)

    else:
        st.info("Upload an image to see the disease detection in action!")

if __name__ == '__main__':
    main()
```EOF

### How to use this updated app:

1.  **Save the code:** Save the provided code as `streamlit_app.py`.
2.  **Create `requirements.txt`:** Create a file named `requirements.txt` in the same directory with the following content:
    ```
    streamlit
    tensorflow
    numpy
    pandas
    matplotlib
    scikit-learn
    opencv-python
    seaborn
    ```
3.  **Run locally:** Open your terminal, navigate to the directory where you saved the files, and run:
    ```bash
    streamlit run streamlit_app.py    "frosty pod rot_00003.png": "Frosty_Pod_Rot",
    "healthy_00003.png": "Healthy",
}

CLASS_NAMES = sorted(list(set(IMAGE_PATHS_AND_LABELS.values())))
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 4
EPOCHS = 10 

# Function to simulate loading the images 
@st.cache_data
def load_image_data_for_training():
    """Generates dummy data for training simulation due to file access restrictions."""
    num_samples = len(IMAGE_PATHS_AND_LABELS)
    
    # Check if there's enough data for all classes in test split (20% of 10 is 2, need at least 1 per class)
    # Since we have very few samples, we'll use a larger test split to get meaningful arrays, but the results remain simulated.
    if num_samples < 5:
        st.error("Insufficient samples for robust training/testing. Using dummy data for demonstration.")
    
    # Generate dummy data for the small set of images
    image_data = np.random.rand(num_samples, IMAGE_SIZE[0], IMAGE_SIZE[1], 3).astype(np.float32)
    labels_list = list(IMAGE_PATHS_AND_LABELS.values())
    label_to_index = {name: i for i, name in enumerate(CLASS_NAMES)}
    label_indices = np.array([label_to_index[label] for label in labels_list])

    # Convert labels to one-hot encoding
    label_one_hot = tf.keras.utils.to_categorical(label_indices, num_classes=NUM_CLASSES)
    return image_data, label_one_hot

# --- 2. Model Definition and Training ---

@st.cache_resource
def build_and_train_model():
    """Builds, compiles, and trains the CNN model."""
    st.info("Loading and processing data...")
    
    # Load simulated data
    X, y = load_image_data_for_training()

    # Split the limited dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=np.argmax(y, axis=1))

    # Define the CNN Model
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', name='last_conv_layer'), # Target layer for Grad-CAM
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    st.info("Training CNN Model... (Using simulated data for demonstration)")
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        verbose=0
    )

    # Evaluate the model
    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    # Generate Classification Report and Confusion Matrix
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Handle case where the split might not contain all classes (due to extremely small dataset)
    try:
        report = classification_report(y_true_classes, y_pred_classes, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    except ValueError:
        report = {'accuracy': 0.0, 'macro avg': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0}}
    
    conf_mat = confusion_matrix(y_true_classes, y_pred_classes)

    return model, history, acc, report, conf_mat, X_test, y_test

# --- 3. Advanced Grad-CAM Implementation ---

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Computes and returns the Grad-CAM heatmap.
    """
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array[np.newaxis, ...])
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8) # Add epsilon for stability
    heatmap = heatmap.numpy()

    return heatmap, preds[0].numpy()

def display_gradcam(img, heatmap, alpha=0.5):
    """
    Overlays the heatmap on the original image and returns a new image (as bytes).
    """
    if 'cv2' not in globals():
        return None, "OpenCV (cv2) is not available to generate the Grad-CAM visualization."

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use OpenCV to resize and apply color map
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
    resized_heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    colormap = cv2.COLORMAP_JET
    heatmap_jet = cv2.applyColorMap(resized_heatmap, colormap)

    # Create the superimposed image
    superimposed_img_bgr = cv2.addWeighted(img_bgr, 1.0 - alpha, heatmap_jet, alpha, 0)

    # Convert back to RGB for Matplotlib/Streamlit display
    superimposed_img_rgb = cv2.cvtColor(superimposed_img_bgr, cv2.COLOR_BGR2RGB)

    # Convert the resulting image to bytes to display in Streamlit
    is_success, buffer = cv2.imencode(".png", cv2.cvtColor(superimposed_img_rgb, cv2.COLOR_RGB2BGR))
    if is_success:
        return buffer.tobytes(), None
    return None, "Failed to encode Grad-CAM image."

def process_uploaded_image(uploaded_file, model):
    """Handles uploaded file, runs prediction, and generates Grad-CAM."""
    
    # 1. Read the image file using Streamlit/Numpy (safer than direct cv2 file read)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
    # Check if cv2 is available for image decoding
    if 'cv2' not in globals():
        st.error("Cannot proceed: OpenCV (cv2) is required for image processing but could not be imported. Please ensure 'opencv-python' is installed.")
        return "N/A", 0.0, None

    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 2. Preprocess the image for the model
    # Note: cv2.resize expects (width, height), Keras expects (height, width) for image size
    img_model_input = cv2.resize(img_rgb, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    img_array = img_model_input.astype('float32') / 255.0

    # 3. Run Grad-CAM
    LAST_CONV_LAYER_NAME = 'last_conv_layer'
    heatmap, preds = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER_NAME)

    # 4. Get prediction result
    predicted_class_index = np.argmax(preds)
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = preds[predicted_class_index] * 100

    # 5. Generate the Grad-CAM visualization
    gradcam_img_bytes, error = display_gradcam(img_rgb, heatmap, alpha=0.5)
    
    if error:
        st.error(error)
        return predicted_class, confidence, None

    return predicted_class, confidence, gradcam_img_bytes

# --- 4. Streamlit Application ---

def main():
    st.set_page_config(layout="wide", page_title="Advanced Disease Detector")

    st.title("🌿 Cacao Disease Detector & Grad-CAM Visualization")
    st.markdown("A Convolutional Neural Network (CNN) for detecting Cacao plant diseases, enhanced with Grad-CAM for model explainability. *(Using simulated training data)*")

    # --- Sidebar for Model Metrics ---
    st.sidebar.title("Model Training & Metrics")
    st.sidebar.info(f"Model trained on **{NUM_CLASSES}** classes: {', '.join(CLASS_NAMES)}")

    # Build and train the model (cached to run only once)
    with st.spinner("Building and training the model... (Simulated to run even with the small dataset)"):
        model, history, acc, report, conf_mat, X_test, y_test = build_and_train_model()
        st.success("Model trained successfully!")

    # Display Model Validation and Accuracy Testing
    st.sidebar.header("Model Validation Results")
    st.sidebar.markdown(f"**Test Set Accuracy:** **{acc:.2f}** (Simulated)")
    
    if history.history:
        st.sidebar.markdown(f"**Test Loss:** **{history.history['val_loss'][-1]:.4f}** (Simulated)")

    with st.sidebar.expander("Detailed Classification Report"):
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)

    if SEABORN_AVAILABLE:
        with st.sidebar.expander("Confusion Matrix"):
            fig, ax = plt.subplots(figsize=(6, 6))
            
            # Predict on the test set for the matrix
            y_pred_classes_cmat = np.argmax(model.predict(X_test, verbose=0), axis=1)
            y_true_classes_cmat = np.argmax(y_test, axis=1)

            cmat_disp = confusion_matrix(y_true_classes_cmat, y_pred_classes_cmat)
            
            sns.heatmap(cmat_disp, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title('Confusion Matrix (Simulated)')
            st.pyplot(fig)
    else:
         st.sidebar.warning("Seaborn is required to display the Confusion Matrix plot.")

    # --- Main App for Prediction ---
    st.header("Upload Image for Prediction")

    uploaded_file = st.file_uploader(
        "Choose a Cacao plant or pod image (JPEG/PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        st.subheader("Results")
        col1, col2 = st.columns(2)

        # Process and Predict
        predicted_class, confidence, gradcam_img_bytes = process_uploaded_image(uploaded_file, model)
        
        # Display Original Image
        with col1:
            st.markdown("<p style='text-align: center; font-size: 1.25rem; font-weight: bold;'>Original Image</p>", unsafe_allow_html=True)
            st.image(uploaded_file, use_column_width=True)
            st.markdown(f"**Predicted Disease:** <span style='color: #1E88E5; font-size: 1.5rem;'>**{predicted_class}**</span>", unsafe_allow_html=True)
            st.markdown(f"**Confidence:** <span style='color: #4CAF50; font-size: 1.5rem;'>**{confidence:.2f}%**</span>", unsafe_allow_html=True)


        # Display Grad-CAM
        with col2:
            st.markdown("<p style='text-align: center; font-size: 1.25rem; font-weight: bold;'>Grad-CAM Visualization</p>", unsafe_allow_html=True)
            if gradcam_img_bytes:
                # Use base64 encoding to display the image bytes in Streamlit
                base64_img = base64.b64encode(gradcam_img_bytes).decode('utf-8')
                st.image(f"data:image/png;base64,{base64_img}", use_column_width=True)
            else:
                st.error("Could not generate Grad-CAM visualization. Ensure OpenCV is installed.")

            st.markdown("""
            **What is Grad-CAM?**
            The Gradient-weighted Class Activation Map (Grad-CAM) highlights the important regions in the image (red/yellow areas) that the model used to make its classification decision.
            """, unsafe_allow_html=True)

    else:
        st.info("Please upload an image to start the disease detection and visualization process.")

if __name__ == '__main__':
    main()IMAGE_SIZE = (128, 128)
BATCH_SIZE = 4
EPOCHS = 10 # Reduced epochs for fast demonstration

# Function to simulate loading the images (in a real app, you'd load from disk)
# Here we use the special content_fetcher structure to access the uploaded files
def load_and_preprocess_image(file_path, label):
    # This is a placeholder for the actual content fetcher logic.
    # In a local environment, this would be a simple tf.io.read_file()
    # For this environment, we use the special content access mechanism.

    # Since we can't execute the content_fetcher API call here,
    # we simulate the data loading by using the accessible file paths and
    # relying on the user to replace this with proper data loading in a real environment.
    # Given the constraint, we will read the images from the file system (or their base64 representation if available)
    # The current execution environment does not allow arbitrary file access, but the
    # files are "accessible." We'll treat the accessible file names as a signal
    # that the image data can be read.

    # --- SIMULATION OF DATA LOADING (Highly dependent on the execution environment) ---
    # Due to limitations in accessing the file contents directly within the Streamlit Python block,
    # we will use the content fetch IDs as placeholders and will rely on the Streamlit
    # file uploader mechanism for user input and a simplified data structure for 'training'.

    # For the actual training data simulation, we must load the images outside of the
    # Streamlit execution loop if possible, or use a cached version.
    try:
        # Load image content (this step requires the actual file content which we simulate)
        # We will use cv2 to read the content via the content fetcher if it were callable here.
        # Since it's not, we'll assume a dummy data structure for simulation purposes.
        # The user's files are accessible, so we must acknowledge them.
        st.cache_data
        def load_image_data_for_training():
            # In a production environment, this function would use
            # tf.keras.utils.image_dataset_from_directory or similar.
            # Here, we generate dummy data based on the required dimensions.
            num_samples = len(IMAGE_PATHS_AND_LABELS)
            image_data = np.random.rand(num_samples, IMAGE_SIZE[0], IMAGE_SIZE[1], 3).astype(np.float32)
            labels_list = list(IMAGE_PATHS_AND_LABELS.values())
            label_to_index = {name: i for i, name in enumerate(CLASS_NAMES)}
            label_indices = np.array([label_to_index[label] for label in labels_list])

            # Convert labels to one-hot encoding
            label_one_hot = tf.keras.utils.to_categorical(label_indices, num_classes=NUM_CLASSES)
            return image_data, label_one_hot

        return load_image_data_for_training()

    except Exception as e:
        # If actual loading fails, fall back to dummy data for demonstration
        st.warning(f"Simulating data due to environment constraints: {e}")
        num_samples = len(IMAGE_PATHS_AND_LABELS)
        image_data = np.random.rand(num_samples, IMAGE_SIZE[0], IMAGE_SIZE[1], 3).astype(np.float32)
        labels_list = list(IMAGE_PATHS_AND_LABELS.values())
        label_to_index = {name: i for i, name in enumerate(CLASS_NAMES)}
        label_indices = np.array([label_to_index[label] for label in labels_list])
        label_one_hot = tf.keras.utils.to_categorical(label_indices, num_classes=NUM_CLASSES)
        return image_data, label_one_hot

# --- 2. Model Definition and Training ---

@st.cache_resource
def build_and_train_model():
    """Builds, compiles, and trains the CNN model."""
    st.info("Loading and processing data...")
    X, y = load_and_preprocess_image("dummy", "dummy") # Load dummy or simulated data

    # Split the limited dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Define the CNN Model
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', name='last_conv_layer'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    st.info("Training CNN Model... (Using dummy data for simulation)")
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        verbose=0 # Turn off verbosity for cleaner Streamlit output
    )

    # Evaluate the model
    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    # Generate Classification Report and Confusion Matrix
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    report = classification_report(y_true_classes, y_pred_classes, target_names=CLASS_NAMES, output_dict=True)
    conf_mat = confusion_matrix(y_true_classes, y_pred_classes)

    return model, history, acc, report, conf_mat, X_test, y_test

# --- 3. Advanced Grad-CAM Implementation (Custom Gradient) ---

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Computes and returns the Grad-CAM heatmap.
    Uses tf.GradientTape for advanced gradient calculation.
    """
    # 1. Get the model's last convolutional layer output and the classifier prediction
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # 2. Compute the gradient of the predicted class with respect to the output of the conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array[np.newaxis, ...])
        if pred_index is None:
            # Find the index of the highest prediction
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # 3. Get the gradient for the predicted class
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # 4. Compute the mean intensity of the gradient over the feature map (Global Average Pooling)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 5. Multiply each channel in the feature map by the corresponding global average gradient
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 6. Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    return heatmap, preds[0].numpy()

def display_gradcam(img, heatmap, alpha=0.4):
    """
    Overlays the heatmap on the original image and returns a new image (as bytes).
    """
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use OpenCV to resize and apply color map
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Grad-CAM usually works better in BGR or grayscale for visualization
    resized_heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    colormap = cv2.COLORMAP_JET # Use JET colormap
    heatmap_jet = cv2.applyColorMap(resized_heatmap, colormap)

    # Create the superimposed image
    superimposed_img_bgr = cv2.addWeighted(img_bgr, 1.0 - alpha, heatmap_jet, alpha, 0)

    # Convert back to RGB for Matplotlib/Streamlit display
    superimposed_img_rgb = cv2.cvtColor(superimposed_img_bgr, cv2.COLOR_BGR2RGB)

    # Convert the resulting image to bytes to display in Streamlit
    is_success, buffer = cv2.imencode(".png", cv2.cvtColor(superimposed_img_rgb, cv2.COLOR_RGB2BGR))
    if is_success:
        return buffer.tobytes()
    return None

def process_uploaded_image(uploaded_file, model):
    """Handles uploaded file, runs prediction, and generates Grad-CAM."""
    # Read the image file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Preprocess the image for the model
    # Note: cv2.resize expects (width, height), Keras expects (height, width) for image size
    img_model_input = cv2.resize(img_rgb, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    img_array = img_model_input.astype('float32') / 255.0

    # Run Grad-CAM
    LAST_CONV_LAYER_NAME = 'last_conv_layer'
    heatmap, preds = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER_NAME)

    # Get prediction result
    predicted_class_index = np.argmax(preds)
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = preds[predicted_class_index] * 100

    # Generate the Grad-CAM visualization
    gradcam_img_bytes = display_gradcam(img_rgb, heatmap, alpha=0.5)

    return predicted_class, confidence, gradcam_img_bytes

# --- 4. Streamlit Application ---

def main():
    st.set_page_config(layout="wide", page_title="Advanced Disease Detector")

    st.title("🌿 Cacao Disease Detector & Grad-CAM Visualization")
    st.markdown("A Convolutional Neural Network (CNN) for detecting Cacao plant diseases, enhanced with Grad-CAM for model explainability.")

    # --- Sidebar for Model Metrics ---
    st.sidebar.title("Model Training & Metrics")
    st.sidebar.info(f"Model trained on **{NUM_CLASSES}** classes: {', '.join(CLASS_NAMES)}")

    # Build and train the model (cached to run only once)
    with st.spinner("Building and training the model... (This is simulated, but the framework is real)"):
        model, history, acc, report, conf_mat, X_test, y_test = build_and_train_model()
        st.success("Model trained successfully!")

    # Display Model Validation and Accuracy Testing
    st.sidebar.header("Model Validation Results")
    st.sidebar.markdown(f"**Test Set Accuracy:** **{acc:.2f}** (Simulated)")
    st.sidebar.markdown(f"**Test Loss:** **{history.history['val_loss'][-1]:.4f}** (Simulated)")

    with st.sidebar.expander("Detailed Classification Report"):
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.highlight_max(axis=0), use_container_width=True)

    with st.sidebar.expander("Confusion Matrix"):
        fig, ax = plt.subplots(figsize=(6, 6))
        cmat_disp = tf.math.confusion_matrix(np.argmax(y_test, axis=1), np.argmax(model.predict(X_test), axis=1)).numpy()
        sns.heatmap(cmat_disp, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix (Simulated)')
        st.pyplot(fig)


    # --- Main App for Prediction ---
    st.header("Upload Image for Prediction")

    uploaded_file = st.file_uploader(
        "Choose a Cacao plant or pod image (JPEG/PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        st.subheader("Results")
        col1, col2 = st.columns(2)

        # Process and Predict
        predicted_class, confidence, gradcam_img_bytes = process_uploaded_image(uploaded_file, model)

        # Display Original Image
        with col1:
            st.markdown("<p style='text-align: center; font-size: 1.25rem; font-weight: bold;'>Original Image</p>", unsafe_allow_html=True)
            st.image(uploaded_file, use_column_width=True)
            st.markdown(f"**Predicted Disease:** <span style='color: #1E88E5; font-size: 1.5rem;'>**{predicted_class}**</span>", unsafe_allow_html=True)
            st.markdown(f"**Confidence:** <span style='color: #4CAF50; font-size: 1.5rem;'>**{confidence:.2f}%**</span>", unsafe_allow_html=True)


        # Display Grad-CAM
        with col2:
            st.markdown("<p style='text-align: center; font-size: 1.25rem; font-weight: bold;'>Grad-CAM Visualization</p>", unsafe_allow_html=True)
            if gradcam_img_bytes:
                # Need to use an image tag with base64 data to display bytes in Streamlit columns smoothly
                base64_img = base64.b64encode(gradcam_img_bytes).decode('utf-8')
                st.image(f"data:image/png;base64,{base64_img}", use_column_width=True)
            else:
                st.error("Could not generate Grad-CAM visualization.")

            st.markdown("""
            **What is Grad-CAM?**
            The Gradient-weighted Class Activation Map (Grad-CAM) technique uses the gradients of the target concept (the predicted class) flowing into the final convolutional layer to produce a coarse localization map highlighting the important regions in the image for predicting the concept.
            The **red/yellow** areas indicate the pixels that *most* contributed to the model making its final decision.
            """, unsafe_allow_html=True)

    else:
        st.info("Please upload an image to start the disease detection and visualization process.")

# Ensure necessary library is imported before the run (Streamlit requires this outside the main function block for modules it needs)
try:
    import seaborn as sns
except ImportError:
    st.error("Seaborn is required for the Confusion Matrix visualization. Please ensure it's installed in your environment.")
    # Fallback to pure matplotlib if seaborn is unavailable
    pass

if __name__ == '__main__':
    main()    # - GDP for 1962
    # - ...
    # - GDP for 2022
    #
    # ...but I want this instead:
    # - Country Name
    # - Country Code
    # - Year
    # - GDP
    #
    # So let's pivot all those year-columns into two: Year and GDP
    gdp_df = raw_gdp_df.melt(
        ['Country Code'],
        [str(x) for x in range(MIN_YEAR, MAX_YEAR + 1)],
        'Year',
        'GDP',
    )
# ... code block
    def load_gdp_data():
        # ... function logic
        gdp_df = pd.read_csv('gdp.csv')
        # ... some processing
        return gdp_df # <- CORRECT: Indented inside the function
# Execution continues here
# Set the title that appears at the top of the page.
'''
# :earth_americas: GDP dashboard

Browse GDP data from the [World Bank Open Data](https://data.worldbank.org/) website. As you'll
notice, the data only goes to 2022 right now, and datapoints for certain years are often missing.
But it's otherwise a great (and did I mention _free_?) source of data.
'''

# Add some spacing
''
''

min_value = gdp_df['Year'].min()
max_value = gdp_df['Year'].max()

from_year, to_year = st.slider(
    'Which years are you interested in?',
    min_value=min_value,
    max_value=max_value,
    value=[min_value, max_value])

countries = gdp_df['Country Code'].unique()

if not len(countries):
    st.warning("Select at least one country")

selected_countries = st.multiselect(
    'Which countries would you like to view?',
    countries,
    ['DEU', 'FRA', 'GBR', 'BRA', 'MEX', 'JPN'])

''
''
''

# Filter the data
filtered_gdp_df = gdp_df[
    (gdp_df['Country Code'].isin(selected_countries))
    & (gdp_df['Year'] <= to_year)
    & (from_year <= gdp_df['Year'])
]

st.header('GDP over time', divider='gray')

''

st.line_chart(
    filtered_gdp_df,
    x='Year',
    y='GDP',
    color='Country Code',
)

''
''


first_year = gdp_df[gdp_df['Year'] == from_year]
last_year = gdp_df[gdp_df['Year'] == to_year]

st.header(f'GDP in {to_year}', divider='gray')

''

cols = st.columns(4)

for i, country in enumerate(selected_countries):
    col = cols[i % len(cols)]

    with col:
        first_gdp = first_year[first_year['Country Code'] == country]['GDP'].iat[0] / 1000000000
        last_gdp = last_year[last_year['Country Code'] == country]['GDP'].iat[0] / 1000000000

        if math.isnan(first_gdp):
            growth = 'n/a'
            delta_color = 'off'
        else:
            growth = f'{last_gdp / first_gdp:,.2f}x'
            delta_color = 'normal'

        st.metric(
            label=f'{country} GDP',
            value=f'{last_gdp:,.0f}B',
            delta=growth,
            delta_color=delta_color
        )
