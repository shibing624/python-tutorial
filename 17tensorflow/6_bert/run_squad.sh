export SQUAD_DIR=/home/users/yanhao/xuming06/nlp/bert_google_models/squad_v1.1
python3 squad_1.1.py \
  --bert_model_dir /home/users/yanhao/xuming06/nlp/bert_google_models/multi_cased_L-12_H-768_A-12  \
  --bert_model_vocab /home/users/yanhao/xuming06/nlp/bert_google_models/multi_cased_L-12_H-768_A-12/vocab.txt \
  --do_train \
  --do_predict \
  --train_file $SQUAD_DIR/train-v1.1.json \
  --predict_file $SQUAD_DIR/dev-v1.1.json \
  --train_batch_size 24 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --fp16 \
  --loss_scale 128 \
  --output_dir /home/users/yanhao/xuming06/nlp/bert_google_models/debug_squad/
