To see my work on this assignment, please review the file “Test_task_Internship_1”

## Description of architectural styles
- CNN: 3 convolutional layers (Conv2D) with MaxPooling — allows for the extraction of local features (lines, corners).
- FFNN: A fully connected network with two hidden layers (128 and 64 neurons) and ReLU.
- RF: 100 decision trees were used.

## Results

| Model | Accuracy|
| -------- | -------- | 
| CNN |0.9886 |
| RF | 0.9719 |
| NN | 0.9763 |

- CNN performs best due to spatial feature extraction
- NN performs well but ignores spatial structure
- RF is simpler and less effective for image data

## Errors
Models often confuse the numbers 4 and 9, or 5 and 6, due to their visual similarity in handwritten text

### CNN errors
<img width="794" height="653" alt="image" src="https://github.com/user-attachments/assets/a02111af-3f64-41e9-a5e3-1401da48209d" />

### RF errors
<img width="794" height="648" alt="image" src="https://github.com/user-attachments/assets/48f30f7e-084e-4dda-a59b-1c286bc74791" />

### NN errors
<img width="794" height="653" alt="image" src="https://github.com/user-attachments/assets/2408b45d-9187-40b8-88f6-03895ae82300" />


