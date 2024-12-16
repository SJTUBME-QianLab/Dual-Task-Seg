
import os

# Performing fine segmentation of pancreatic cancer based on the coarse segmentation results obtained from the pancreas mask.
os.system(
    "python evaluate_tumor_fine_based_on_mask.py "
    "--model_name Tumor_3D_Fine_Fine_renji,rmyy.pth "
    "--path Tumor_3D_Fine_Fine_On_renji,rmyy_time "
    "--coarse_path Tumor_3D_Fine_On_renji,rmyy_time "
    "--thresh 0.6 "
    "--data msd "
)
os.system(
    "python evaluate_tumor_fine_based_on_mask.py "
    "--model_name Tumor_3D_Fine_Fine_msd,rmyy.pth "
    "--path Tumor_3D_Fine_Fine_On_msd,rmyy_time "
    "--coarse_path Tumor_3D_Fine_On_msd,rmyy_time "
    "--thresh 0.1 "
    "--data renji "
)
os.system(
    "python evaluate_tumor_fine_based_on_mask.py "
    "--model_name Tumor_3D_Fine_Fine_msd,renji.pth "
    "--path Tumor_3D_Fine_Fine_On_msd,renji_time "
    "--coarse_path Tumor_3D_Fine_On_msd,renji_time "
    "--thresh 0.9 "
    "--data rmyy "
)

# Performing fine segmentation of pancreatic cancer based on the coarse segmentation results of pancreatic cancer obtained from the coarse pancreas segmentation,
# where --coarse_path denotes the folder for the pancreatic fine segmentation model.
os.system(
    "python evaluate_tumor_fine_based_pancreas_seg.py "
    "--model_name Tumor_3D_Fine_Fine_renji,rmyy.pth "
    "--path Tumor_3D_Fine_Fine_On_renji,rmyy_time "
    "--coarse_path Tumor_3D_Fine_On_renji,rmyy_time "
    "--thresh 0.7 "
    "--data msd "
)
os.system(
    "python evaluate_tumor_fine_based_pancreas_seg.py "
    "--model_name Tumor_3D_Fine_Fine_msd,rmyy.pth "
    "--path Tumor_3D_Fine_Fine_On_msd,rmyy_time "
    "--coarse_path Tumor_3D_Fine_On_msd,rmyy_time "
    "--thresh 0.1 "
    "--data renji "
)
os.system(
    "python evaluate_tumor_fine_based_pancreas_seg.py "
    "--model_name Tumor_3D_Fine_Fine_msd,renji.pth "
    "--path Tumor_3D_Fine_Fine_On_msd,renji_time "
    "--coarse_path Tumor_3D_Fine_On_msd,renji_time "
    "--thresh 0.9 "
    "--data rmyy "
)
