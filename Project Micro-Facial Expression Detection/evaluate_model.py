import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os

def augment_image(img):
    """Simple data augmentation"""
    augmented = []
    
    # Original
    augmented.append(img)
    
    # Horizontal flip
    augmented.append(cv2.flip(img, 1))
    
    # Slight rotation
    rows, cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 5, 1)
    rotated = cv2.warpAffine(img, M, (cols, rows))
    augmented.append(rotated)
    
    return augmented

def load_data():
    """Load and preprocess data"""
    X, y = [], []
    
    # Load Truth data (0)
    truth_dir = 'data/micro/train/truth'
    for filename in os.listdir(truth_dir):
        if filename.endswith('.jpg'):
            img_path = os.path.join(truth_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, (48, 48))
                # Apply augmentation
                augmented_imgs = augment_image(img_resized)
                for aug_img in augmented_imgs:
                    X.append(aug_img.flatten())
                    y.append(0)  # Truth
    
    # Load Lie data (1)
    lie_dir = 'data/micro/train/lie'
    for filename in os.listdir(lie_dir):
        if filename.endswith('.jpg'):
            img_path = os.path.join(lie_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, (48, 48))
                # Apply augmentation
                augmented_imgs = augment_image(img_resized)
                for aug_img in augmented_imgs:
                    X.append(aug_img.flatten())
                    y.append(1)  # Lie
    
    # Shuffle data after augmentation
    X, y = shuffle(np.array(X), np.array(y), random_state=42)
    return X, y

def evaluate_model():
    """Evaluate model and create confusion matrix"""
    print("Loading data...")
    X, y = load_data()
    
    print(f"Loaded {len(X)} images")
    print(f"Truth samples: {np.sum(y == 0)}")
    print(f"Lie samples: {np.sum(y == 1)}")
    
    # Normalize data
    X = X.astype('float32') / 255.0
    
    # Split data (same as training)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Load trained model
    try:
        with open('micro_model_simple.pkl', 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully!")
    except:
        print("No trained model found. Run simple_train.py first.")
        return
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Truth', 'Lie'], 
                yticklabels=['Truth', 'Lie'])
    plt.title('Confusion Matrix - Micro-Facial Expression Detection')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    # Create images directory if not exists
    os.makedirs('images', exist_ok=True)
    
    # Save confusion matrix
    plt.savefig('images/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed metrics
    print("\n" + "="*50)
    print("CONFUSION MATRIX")
    print("="*50)
    print(f"Test samples: {len(y_test)}")
    print("\nConfusion Matrix:")
    print(f"{'':15} {'Truth':>8} {'Lie':>8} {'Total':>8}")
    print(f"{'Truth':15} {cm[0,0]:>8} {cm[0,1]:>8} {cm[0,0]+cm[0,1]:>8}")
    print(f"{'Lie':15} {cm[1,0]:>8} {cm[1,1]:>8} {cm[1,0]+cm[1,1]:>8}")
    print(f"{'Total':15} {cm[0,0]+cm[1,0]:>8} {cm[0,1]+cm[1,1]:>8} {len(y_test):>8}")
    
    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    
    precision_truth = tn / (tn + fn) if (tn + fn) > 0 else 0
    precision_lie = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_truth = tn / (tn + fp) if (tn + fp) > 0 else 0
    recall_lie = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_truth = 2 * (precision_truth * recall_truth) / (precision_truth + recall_truth) if (precision_truth + recall_truth) > 0 else 0
    f1_lie = 2 * (precision_lie * recall_lie) / (precision_lie + recall_lie) if (precision_lie + recall_lie) > 0 else 0
    
    print(f"\nMETRICS:")
    print(f"Precision (Truth): {tn}/{tn+fn} = {precision_truth:.1%}")
    print(f"Precision (Lie): {tp}/{tp+fp} = {precision_lie:.1%}")
    print(f"Recall (Truth): {tn}/{tn+fp} = {recall_truth:.1%}")
    print(f"Recall (Lie): {tp}/{tp+fn} = {recall_lie:.1%}")
    print(f"F1-Score (Truth): {f1_truth:.1%}")
    print(f"F1-Score (Lie): {f1_lie:.1%}")
    
    # Overall accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    print(f"\nOverall Test Accuracy: {accuracy:.1%}")
    
    # Classification report
    print("\n" + "="*50)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y_test, y_pred, target_names=['Truth', 'Lie']))

if __name__ == "__main__":
    evaluate_model()