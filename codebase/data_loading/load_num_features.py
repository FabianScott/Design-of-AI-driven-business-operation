import pickle

feature_savename = "data/features_per_num.pkl"
accs_savename = "data/accuracies_per_num.pkl"

def save_features_per_num(features=None, accuracies=None, feature_savename=feature_savename, accs_savename=accs_savename):
    if features is not None:
        with open(feature_savename, "wb") as f:
            pickle.dump(features, f)
    if accuracies is not None:
        with open(accs_savename, "wb") as f:
            pickle.dump(accuracies, f)

def load_features_per_num(feature_savename=feature_savename, accs_savename=accs_savename):
    try:
        with open(feature_savename, "rb") as f:
            features = pickle.load(f)
    except FileNotFoundError:
        features = None
        print("Feature file not found.")
    try:
        with open(accs_savename, "rb") as f:
            accuracies = pickle.load(f)
    except FileNotFoundError:
        accuracies = None
        print("Accuracy file not found.")

    return features, accuracies
