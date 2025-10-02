import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.utils import shuffle

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

def train_model():
    """Train simple ML model"""
    print("Loading data...")
    X, y = load_data()
    
    print(f"Loaded {len(X)} images")
    print(f"Truth samples: {np.sum(y == 0)}")
    print(f"Lie samples: {np.sum(y == 1)}")
    
    # Normalize data
    X = X.astype('float32') / 255.0
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training model...")
    # Use RandomForest with reduced complexity to prevent overfitting
    model = RandomForestClassifier(
        n_estimators=50,      # Giảm từ 100 xuống 50
        max_depth=10,         # Giới hạn độ sâu cây
        min_samples_split=10, # Tăng min samples để split
        min_samples_leaf=5,   # Tăng min samples ở leaf
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    print(f"Training accuracy: {train_acc:.3f}")
    print(f"Test accuracy: {test_acc:.3f}")
    
    # Save model
    with open('micro_model_simple.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Model saved as 'micro_model_simple.pkl'")
    return model

if __name__ == "__main__":
    train_model()