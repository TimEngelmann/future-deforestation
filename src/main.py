import os

from features.build_features import build_features
from models.train_model import train_model
from models.predict_model import predict_model
from visualization.visualize import visualize

if __name__ == "__main__":
    output_px = 40
    input_px = 400
    # build_features(output_px, input_px)

    init_nr = None
    max_epochs = 1
    train_model(init_nr, max_epochs, output_px, input_px)

    model_nr = [model_folder.split("_")[-1] for model_folder in os.listdir(f"lightning_logs/") if "version_" in model_folder]
    predict_model(model_nr, output_px, input_px)
    visualize(model_nr, output_px, input_px)
