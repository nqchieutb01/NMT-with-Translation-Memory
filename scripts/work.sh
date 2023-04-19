set -e

ckpt=${MTPATH}/mt.ckpts/envi/ckpt.exp.dynamic/epoch29_batch88999_devbleu67.66_testbleu67.16
dataset=${MTPATH}/envi
python3 work.py --load_path ${ckpt} \
       --test_data ${dataset}/test.txt \
       --output_path ${dataset}/test.out.fixed.txt \
       --comp_bleu
