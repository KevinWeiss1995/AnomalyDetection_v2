import tensorflow as tf
from tensorflow.keras import backend as K

def focal_loss(gamma=2., alpha=.25):
    """
    Creates a focal loss function for training deep neural networks with imbalanced datasets.
    
    Focal loss reduces the relative loss for well-classified examples and focuses more on
    hard, misclassified examples. Useful when some examples are easy to classify while
    others are hard.
    
    Args:
        gamma: Focus parameter that reduces the loss for well-classified examples.
              Higher values mean more focus on hard examples. Default is 2.
        alpha: Weight parameter for class imbalance. Default is 0.25.
    
    Returns:
        A loss function that can be used in model.compile()
    """
    def focal_loss_fixed(y_true, y_pred):
        """Keras loss function accepts y_true and y_pred"""
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1 + K.epsilon()) +
                      (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
    return focal_loss_fixed
