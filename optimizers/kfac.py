import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

@tf.function(reduce_retracing=True)
def kfac_update(model, grads, accumulated_grads, fisher_matrices, momentum, damping, learning_rate):
    """KFAC update step with fixed input signatures to prevent retracing"""
    if grads is None:
        return accumulated_grads, fisher_matrices
        
    for layer, grad in zip(model.trainable_variables, grads):
        if isinstance(layer, tf.Variable):
            layer_name = layer.name
            
           
            accumulated_grads[layer_name] = (
                momentum * accumulated_grads[layer_name] + 
                (1 - momentum) * grad
            )
            
            fisher_update = tf.matmul(tf.expand_dims(grad, 1), tf.expand_dims(grad, 0))
            fisher_matrices[layer_name] = (
                fisher_matrices[layer_name] + 
                damping * fisher_update
            )
            
            damping_eye = damping * tf.eye(tf.shape(fisher_update)[0], dtype=grad.dtype)
            fisher_damped = fisher_matrices[layer_name] + damping_eye
            fisher_inv = tf.linalg.inv(fisher_damped)
            nat_grad = tf.matmul(fisher_inv, tf.reshape(accumulated_grads[layer_name], (-1, 1)))
            
            layer.assign_sub(learning_rate * tf.reshape(nat_grad, grad.shape))
    
    return accumulated_grads, fisher_matrices

class KFACCallback(keras.callbacks.Callback):
    def __init__(self, damping=0.001, momentum=0.9):
        super().__init__()
        self.damping = damping
        self.momentum = momentum
        self.accumulated_grads = {}
        self.fisher_matrices = {}
        
    def on_train_begin(self, logs=None):
        """Initialize optimizer state at the start of training"""
        for var in self.model.trainable_variables:
            self.accumulated_grads[var.name] = tf.zeros_like(var)
            shape = var.shape[0]
            self.fisher_matrices[var.name] = tf.eye(shape, dtype=var.dtype)
    
    def on_train_batch_begin(self, batch, logs=None):
        """Ensure variables exist for new layers if model structure changes"""
        for var in self.model.trainable_variables:
            if var.name not in self.accumulated_grads:
                self.accumulated_grads[var.name] = tf.zeros_like(var)
                shape = var.shape[0]
                self.fisher_matrices[var.name] = tf.eye(shape, dtype=var.dtype)

    def on_train_batch_end(self, batch, logs=None):
        """Apply KFAC updates after gradient step"""
        if not hasattr(self.model, 'optimizer'):
            return
      
        x_batch = logs.get('x')
        y_batch = logs.get('y')
        
        if x_batch is None or y_batch is None:
            return
            
        with tf.GradientTape() as tape:
            predictions = self.model(x_batch, training=True)
            loss = self.model.compiled_loss(y_batch, predictions)
        
        grads = tape.gradient(loss, self.model.trainable_variables)

        self.accumulated_grads, self.fisher_matrices = kfac_update(
            self.model,
            grads,
            self.accumulated_grads,
            self.fisher_matrices,
            tf.constant(self.momentum, dtype=tf.float32),
            tf.constant(self.damping, dtype=tf.float32),
            tf.constant(self.model.optimizer.learning_rate, dtype=tf.float32)
        )
