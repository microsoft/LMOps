
python generate_dense_embeddings.py model_file=/media/disk1/ohadr/lr1e-5/dpr_biencoder.29 \
	ctx_src=dpr_grail shard_id=0 num_shards=1 out_file=/mnt/netapp7/ohadr/GrailSmBop/DPR/entities_c29_lr1_enc

python dense_retriever.py model_file=/media/disk1/ohadr/lr1e-5/dpr_biencoder.29 qa_dataset=grailqa_train ctx_datatsets=[dpr_grail] \
 encoded_ctx_files=["/mnt/netapp7/ohadr/GrailSmBop/DPR/entities_c29_lr1_enc_*"] out_file=/mnt/netapp7/ohadr/GrailSmBop/DPR/dpr_pred_train_c29_lr1.json
python dense_retriever.py model_file=/media/disk1/ohadr/lr1e-5/dpr_biencoder.29 qa_dataset=grailqa_dev ctx_datatsets=[dpr_grail] \
  encoded_ctx_files=["/mnt/netapp7/ohadr/GrailSmBop/DPR/entities_c29_lr1_enc_*"] out_file=/mnt/netapp7/ohadr/GrailSmBop/DPR/dpr_pred_dev_c29_lr1.json


python dpr/data/download_data.py --resource data.retriever.qas.trivia-dev
	[optional --output_dir {your location}]


	(grail) ohadr@pc-jonathan-g01:~/GrailSmBop/DPR$ head dpr_pred_dev.json -n 1000 |grep --color=always -e "^" -e  true
