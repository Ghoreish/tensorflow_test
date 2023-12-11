import tensorflow as tf

# Define your input data (list of lists)
input_data = []
output_data = []
l = []
f = open('data.txt', 'r')
data = f.readline().strip()
while data:
    x, y = list(map(int, data.split('\t')))
    res = (x - y) / y * 100
    l.append(res)
    data = f.readline().strip()
f.close()
l = l[::-1]
ll = len(l)

for i in range(20, ll):
    input_data.append(l[i - 20: i])
    output_data.append(l[i])

# Define your corresponding output data (list of numbers)

# Convert data to TensorFlow tensors
X_train = tf.constant(input_data, dtype=tf.float32)
y_train = tf.constant(output_data, dtype=tf.float32)

# Define a simple neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(len(input_data[0]),)),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100000, verbose=0)  # Adjust the number of epochs as needed
f = open("test.txt", "r")
l2 = []
data = f.readline().strip()
while data:
    x, y = list(map(int, data.split('\t')))
    res = (x - y) / y * 100
    l2.append(res)
    data = f.readline().strip()
f.close()
l2 = l2[::-1]

# Now, let's make predictions for a new set of data
new_data = l2
new_data = tf.constant([l2], dtype=tf.float32)
predictions = model.predict(new_data)

print(f'Predicted Output: {predictions[0, 0]}')
