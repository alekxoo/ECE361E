import re
import matplotlib.pyplot as plt 
import numpy as np

# Read log file content (assuming it's in a string called log_content)
with open('out.txt', 'r') as f:
    log_content = f.read()

# # Extract epoch numbers and train loss
pattern = r"Epoch: \[(\d+)/25\], Step: \[400/468\], Loss: ([\d.]+) Acc: ([\d.]+)%\nTest accuracy: ([\d.]+) % Test loss: ([\d.]+)"

matches = re.findall(pattern, log_content)
epochs = [int(m[0]) for m in matches]
train_loss = [float(m[1]) for m in matches]
train_accuracy = [float(m[2]) for m in matches]
test_accuracy = [float(m[3]) for m in matches]
test_loss = [float(m[4]) for m in matches]

plt.figure(figsize=(10,6))
plt.plot(epochs, train_loss, 'b-', label = 'Train Loss')
plt.plot(epochs, test_loss, 'r-', label = 'Test Loss')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Plot for 0.8 Dropout Probability")
plt.grid(True, linestyle='--', alpha = 0.7)
plt.legend()

plt.savefig('loss_plot.png')    


plt.clf()
plt.figure(figsize = (10,6))
plt.plot(epochs, train_accuracy, 'b-', label = "Train Accuracy")
plt.plot(epochs, test_accuracy, 'r-', label = "Test Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Plot for 0.8 Dropout Probability")
plt.grid(True, linestyle='--', alpha = 0.7)
plt.legend()
plt.savefig('accuracy_plot.png')