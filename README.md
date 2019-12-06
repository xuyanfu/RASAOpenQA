# RASAOpenQA
Code for the EMNLP 2019 paper "[Ranking and Sampling in Open-Domain Question Answering](https://www.aclweb.org/anthology/D19-1245/)"

# Requirements
- Python (>=3.5.6)
- TensorFlow (=1.8.0)

# Preprocess
	cd Ranker
	mkdir data
	cd data 
	
download [embeddings](https://pan.baidu.com/s/1_D1voXnCPVNgrDTvZRaohw), [datasets](https://pan.baidu.com/s/1-BZdTgixRXRC54Peh7GpBA) and [corenlp](https://pan.baidu.com/s/1dvraJlIOjWFvX8mbCPtGkw)

	unzip embeddings.zip
	unzip datasets.zip
	unzip corenlp.zip


# Ranker
	cd Ranker
	mkdir tmp_data
	mkdir models
	python3 initvim_anas.py #Initialize
	python3 run.py #Train & Evaluate

	
# Reader
	export PYTHONPATH=${PYTHONPATH}:'Path_to_Reader'
	cd Reader
	mkdir probs
	mkdir result
	mkdir tmp_data
	cd ../Ranker/tmp_data/
	cp  list_* id2scores_* ../../Reader/tmp_data
	cd ../Reader
	python3 run.py merge result/model #Train & Evaluate
	python3 docqa/eval/triviaqa_full_document_eval.py  --step 110  -c open-dev  --rank 1  --n_paragraphs 30  --shuffle 0   --max_answer_len 8 -o question-output.json -p paragraph-output.csv result/model-date-time #Test MAX Method
	python3 docqa/eval/init_data.py #Test SUM Method
	
# Remark
The above commands are used to train and test on the Quasar-T dataset. 
You can download the preprocessed data for [SearchQA](https://pan.baidu.com/s/1BYm6hESUeiWVqfja7w7Yww) and [TriviaQA](https://pan.baidu.com/s/1JNOyL83hcYSZpLijA_Gt6g) and save them in Ranker/tmp_data/ directory.
Some parameters of the model should be changed according to the paper. 
After that, you can train the Ranker and the Reader for SearchQA and TriviaQA.

	
	
	

	
