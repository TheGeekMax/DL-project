import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def generate_graphs_from_model(history, model_type, test_x, test_y, y_pred):
    if not( history is None):
        plt.plot(history.history["loss"], label="train_loss")
        plt.plot(history.history["val_loss"], label="validation_loss")
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"graphs/{model_type}_loss.png")

        plt.clf()
        plt.plot(history.history["accuracy"], label="train_accuracy")
        plt.plot(history.history["val_accuracy"], label="validation_accuracy")
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(f"graphs/{model_type}_accuracy.png")

    # step 5.2 : generate the confusion matrix
    cm = confusion_matrix(test_y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Earthquake", "Earthquake"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(f"graphs/{model_type}_confusion_matrix.png")
    plt.clf()