cd DiFE

python main.py --combine_type $combine_type --model_name 'wietsedv/bert-base-dutch-cased'
python main.py --combine_type $combine_type --model_name 'wietsedv/bert-base-dutch-cased' --fine_tuning
python main.py --combine_type $combine_type --model_name 'wietsedv/bert-base-dutch-cased-finetuned-sentiment'
