import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Input, Subtract, Activation
from keras import backend
import matplotlib.pyplot as plt
import seaborn as sns
import csv

INPUT_DIM = 3

# Model.
h_1 = Dense(128, activation="relu")
h_2 = Dense(64, activation="relu")
h_3 = Dense(32, activation="relu")
s = Dense(1)

# Relevant document score.
rel_doc = Input(shape=(INPUT_DIM,))
h_1_rel = h_1(rel_doc)
h_2_rel = h_2(h_1_rel)
h_3_rel = h_3(h_2_rel)
rel_score = s(h_3_rel)

# Irrelevant document score.
irr_doc = Input(shape=(INPUT_DIM,))
h_1_irr = h_1(irr_doc)
h_2_irr = h_2(h_1_irr)
h_3_irr = h_3(h_2_irr)
irr_score = s(h_3_irr)

# Subtract scores.
diff = Subtract()([rel_score, irr_score])

# Pass difference through sigmoid function.
prob = Activation("sigmoid")(diff)

# Build model.
model = Model(inputs=[rel_doc, irr_doc], outputs=prob)
optimizer = keras.optimizers.SGD(learning_rate=0.1)
loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)

model.summary()

# Generate scores from document/query features.
get_score = backend.function([rel_doc], [rel_score])

with open("train.csv","r") as file_:
    reader = csv.reader(file_)
    train_input = list(reader)
    train_output = [x.pop(-1) for x in train_input]
    
train_input = np.array(train_input).astype('float32')
train_output = np.array(train_output).astype('float32')

def ndcg(y_true, y_score, k=20):
    y_true = y_true.ravel()
    y_score = y_score.ravel()
    y_true_sorted = sorted(y_true, reverse=True)
    ideal_dcg = 0
    for i in range(k):
        ideal_dcg += (2 ** y_true_sorted[i] - 1.) / np.log2(i + 2)
    dcg = 0
    argsort_indices = np.argsort(-y_score)
    for i in range(k):
        dcg += (2 ** y_true[argsort_indices[i]] - 1.) / np.log2(i + 2)
    ndcg = dcg / ideal_dcg
    return ndcg

train_ndcgs = []

epochs = 20
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch+1,))
    # Running the forward and back propagation to the number of rows times in every epoch
    for _ in range(np.shape(train_input)[0]):
        # Open a GradientTape to record the operations run
        # during the forward pass, which enables autodifferentiation.
        with tf.GradientTape() as tape:

            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            i=0
            j=0
            while i==j:
                i, j = np.random.randint(np.shape(train_input)[0], size=2)
            #print("\ni : %d, j : %d" % (i,j))
            x1 = np.reshape(train_input[i], (1, INPUT_DIM))
            x2 = np.reshape(train_input[j], (1, INPUT_DIM))
            y_pred = model([x1, x2], training=True)  # y_pred for this document

            y_actual = 1 if train_output[i]>train_output[j] else 0
            y_actual = np.reshape(y_actual, (1,1))
            # Compute the loss value for this minibatch.
            loss_value = loss_fn(y_actual, y_pred)

        #print("\nLoss: " + str(loss_value))

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
    
    if epoch%5 == 0 or epoch == 0:
        ranked_list = get_score(train_input)[0]
        train_metric = ndcg(train_output, ranked_list)
        train_ndcgs.append(train_metric)
        print("Epoch: {}".format(epoch+1))
        print("NDCG@20 | train: {}".format(
                train_ndcgs))

print("\nRanked list")
ranked = (get_score(train_input)[0]).ravel()
ranked = np.argsort(-ranked)
print(train_input[ranked])
sns.set_context("poster")
plt.plot(train_ndcgs, label="Train")
plt.legend(loc="best")
plt.xlabel("Epoch")
plt.ylabel("NDCG@20")
plt.ylim(0, 1.1)
plt.tight_layout()
plt.show()
