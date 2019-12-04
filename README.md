#RASAOpenQA
Code for the EMNLP 2019 paper "Ranking and Sampling in Open-Domain Question Answering"

# Requirements
- Python (>=3.5.6)
- TensorFlow (=1.8.0)

#Preprocess


#Ranker
	cd Ranker
	mkdir tmp_data
	mkdir data
	mkdir models
	python3 init_anas.py #Initialize
	python3 run.py #Train & Evaluate

	
#Reader

	cd Reader
	mkdir probs
	mkdir result
	mkdir tmp_data
	cd ../Ranker/tmp_data/
	cp  list_* id2scores_* ../../Reader/tmp_data
	cd ../Reader
	python3 run.py merge result/model
	
	

	
