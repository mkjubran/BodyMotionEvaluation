#export PYTHONPATH=$PYTHONPATH:/AIARUPD/BodyMotionEvaluation/bpe
python3 ./bin/train_bpe.py --data_dir ../SARA_released/ --use_footvel_loss --logdir logdir
