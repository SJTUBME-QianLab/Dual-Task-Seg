
import os

# --model_name represents the name of the model, --path denotes the folder name, --data indicates the dataset to be tested, and --coarse_path represents the folder for the coarse segmentation model.
os.system(
    "python evaluate_pancreas_fine.py --data renji --model_name Generalize_Pancreas_Fine_renji.pth --path Fine_Pancreas_Generalize_renji_time --coarse_path Coarse_Pancreas_Generalize_renji_time"
)
os.system(
    "python evaluate_pancreas_fine.py --data rmyy --model_name Generalize_Pancreas_Fine_rmyy.pth --path Fine_Pancreas_Generalize_rmyy_time --coarse_path Coarse_Pancreas_Generalize_rmyy_time"
)
os.system(
    "python evaluate_pancreas_fine.py --data rmyy --model_name Generalize_Pancreas_Fine_rmyy.pth --path Fine_Pancreas_Generalize_msd_time --coarse_path Coarse_Pancreas_Generalize_msd_time"
)
