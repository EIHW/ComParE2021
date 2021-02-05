for combine_type in mean sum
do
  python main.py --combine_type $combine_type --model_name 'wietsedv/bert-base-dutch-cased'
done

for combine_type in mean sum
do
  python main.py --combine_type $combine_type --model_name 'wietsedv/bert-base-dutch-cased' --fine_tuning
done

for combine_type in mean sum
do
  python main.py --combine_type $combine_type --model_name 'wietsedv/bert-base-dutch-cased-finetuned-sentiment'
done