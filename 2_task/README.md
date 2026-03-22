## Loading the dataset

In the 'Test_task_2' file, you can see my dataset and testing the model

## Note
GitHub may not display the notebook correctly due to metadata issues from Google Colab.  
To view the notebook with results, please download the `.ipynb` file and open it locally in Jupyter Notebook or Google Colab.

I used the Animals-10 dataset from Kaggle. Number of classes: 10 (dog, horse, elephant, butterfly, chicken, cat, cow, sheep, spider, squirrel).

During the analysis of the Animals-10 dataset, I observed a significant difference in the number of images between classes.

<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/312f1086-34db-4ac0-94dc-e7c280ced3b5" />

The model is better at recognizing dogs and spiders because it has seen them about three times more often than elephants.

Examples of photos:
<img width="1189" height="543" alt="image" src="https://github.com/user-attachments/assets/7edb5c76-ec9a-41ae-9a59-a99ec4f1e466" />

## Technical implementation
1. NLP Module (NER)
- Architecture: distilbert-base-uncased (fine-tuned)
- Approach: A custom dataset was created with over 20 sentence patterns to train the model to identify animal entities in context

2. CV Module (Image Classification)
- Architecture: Custom CNN with 4 convolutional layers
- Use of BatchNorm and Dropout to stabilize the model and prevent overfitting
- OneCycleLR scheduler for fast convergence
- Input image size: 128x128

The training process was conducted over 10 epochs using a OneCycleLR scheduler to optimize the learning rate.

<img width="1200" height="400" alt="Figure_1" src="https://github.com/user-attachments/assets/c12e5570-69c9-4eb9-a03b-a2155bd6f5df" />

The model achieved a final validation accuracy of approximately 78-80%.


## Pipeline
How the pipeline.py script works:
- It takes text and image_path as input
- The NER model determines which animal to search for
- The CNN model identifies the animal in the photo
- The system compares the results and returns True or False


### The following files must be present in the project directory:
- pipeline.py
- NER_train.py
- NER_inference.py
- Classification_train.py 
- Classification_inference.py
- animal_cnn.pth 
- idx_to_class.json
- ner_model/
 
### Input Data
- A text describing an animal
- An image file (e.g., .jpg, .png)

  ### To start type:
  ```bash
  python pipeline.py --text "It's a (name of animal)" --image "your_image.jpg"


## Demo

The notebook with results and pipeline examples is included in this folder.
If GitHub does not display it correctly, please download the notebook and open it locally.
  




