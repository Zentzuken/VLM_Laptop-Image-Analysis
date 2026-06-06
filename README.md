Brian Abraham Jordan Suntpiet 2310101005


How to run project + instructions and details, 
this can be done right after downloaded (unless something breaks)

- Insert image and rename to 'query.jpg'
- Run 'search_vlm_ollama.py'


Evaluating model performance (ground truth)
- ensure ground truth dataset is set-up
- Run 'evaluate_ground_truth.py'


How to use project

- download all dependancies, project was built with 'python 3.11'
- Run scrapper to add more dataset
- run 'test_dataset.py' to check if dataset has missing values

- - Optional run 'train_clip.py' if wants to use the preset fine-tune with a higher epoch value

- run 'rebuild_embeddings.py' 
- - re-run whenever changing models, makes sure to switch to desired model inside of the script

- run 'build_faiss.py'


