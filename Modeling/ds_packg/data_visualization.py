import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

def visualize_error(train_errors, valid_errors, val_epochs):
    plt.plot(range(len(train_errors)), train_errors, label='Training Error')
    plt.plot(val_epochs, valid_errors, label='Validation Error')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.ylim(-0.002,0.02)
    plt.show()
    
def viz_scatter_r2(val_outcomes, val_pred):
    R2 = r2_score(val_outcomes, val_pred)
    RMSE = mean_squared_error(val_outcomes, val_pred, squared = False)
    
    plt.scatter(val_outcomes, val_pred)
    plt.xlabel("Correct Outcome")
    plt.ylabel("Predicted Outcome")
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.title(f"R2 Score: {R2:.3f}; \
                RMSE: {RMSE:.4f}")
    plt.show()
