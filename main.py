import os
import tensorflow as tf
from utils.git import get_git_repo_root
from data.loader import load_data
from training import train_with_cross_validation, train_final_model
from losses.focal_loss import focal_loss
from sklearn.metrics import classification_report

def main():
   
    print("Loading data...")
    X, y, X_test, y_test, feature_names = load_data()
    
    fold_scores = train_with_cross_validation(X, y)

    print("\nTraining final model...")
    final_model = train_final_model(X, y, X_test, y_test)

    model_dir = os.path.join(get_git_repo_root(), 'results', 'models', 'network')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'model.keras')
    
    final_model.save(model_path)
    print(f"\nModel saved to: {model_path}")

    loaded_model = tf.keras.models.load_model(model_path)
    test_predictions = (loaded_model.predict(X_test) > 0.5).astype(int)
    print("\nVerification of saved model:")
    print(classification_report(y_test, test_predictions))


if __name__ == "__main__":
    main()