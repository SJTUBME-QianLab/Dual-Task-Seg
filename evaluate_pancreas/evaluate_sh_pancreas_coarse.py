import os

# The model_name represents the name of the model, path denotes the folder name, and data indicates the dataset to be tested.
os.system(
    "python evaluate_pancreas_coarse.py --data renji --model_name Generalize_Pancreas_Coarse_renji.pth --path Coarse_Pancreas_Generalize_renji_time"
)
os.system(
    "python evaluate_pancreas_coarse.py --data rmyy --model_name Generalize_Pancreas_Coarse_rmyy.pth --path Coarse_Pancreas_Generalize_rmyy_time"
)
os.system(
    "python evaluate_pancreas_coarse.py --data msd --model_name Generalize_Pancreas_Coarse_msd.pth --path Coarse_Pancreas_Generalize_msd_time"
)
