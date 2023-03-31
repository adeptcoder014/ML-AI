import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
# ===============================================

# -------------- BUILDING_MODEL  -----------------------------------------------


def build_model(my_learning_rate):
    ''' Create and compile a simple linear regression model 👌'''
    # Most simple tf.keras models are sequential.
    # A sequential model contains one or more layers.

    model = tf.keras.models.Sequential()  # Instantiation

    model.add(tf.keras.layers.Dense(
        units=1,
        input_shape=(1,)
    ))  # ---------------------------- adding neural layers
    model.compile(
        optimizer=tf.keras.optimizers.experimental.RMSprop(
            learning_rate=my_learning_rate),
        loss='mean_squared_error',
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )  # ------------------------------ compiling the model
    return model

# -------------- TRAINING_MODEL  -----------------------------------------------


def train_model(model, feature, label, epochs, batch_size):
    ''' Train the model by feeding it the data'''
    # Feed the feature values and the label values to the
    # model. The model will train for the specified number
    # of epochs, gradually learning how the feature values
    # relate to the label values.
    history = model.fit(
        x=feature,
        y=label,
        batch_size=batch_size,
        epochs=epochs
    )
    # print("History ========================================", history.history)
    # Gather the trained model's weight and bias.
    trained_weight = model.get_weights()[0]
    trained_bias = model.get_weights()[1]
    # The list of epochs is stored separately from the
    # rest of history.
    epochs = history.epoch
    # Gather the history (a snapshot) of each epoch.
    hist = pd.DataFrame(history.history)  # ----------------> {loss:[],rmse:[]}
    # Specifically gather the model's root mean
    # squared error at each epoch.
    rmse = hist["root_mean_squared_error"]
    loss = hist["loss"]

    return trained_weight, trained_bias, epochs, rmse, loss

# -------------- PLOTTING_MODEL  -----------------------------------------------


def plot_the_model(trained_weight, trained_bias, feature, label):
    """Plot the trained model against the training feature and label."""
    # Label the axes.
    plt.xlabel("feature")
    plt.ylabel("label")

    # Plot the feature values vs. label values.
    plt.scatter(feature, label)
    # Create a red line representing the model. The red line starts
    # at coordinates (x0, y0) and ends at coordinates (x1, y1).
    x0 = 0
    y0 = trained_bias
    x1 = feature[-1]
    y1 = trained_bias + (trained_weight * x1)
    plt.plot([x0, x1], [y0, y1], c='r')
    # Render the scatter plot and the red line.
    plt.show()

# -------------- PLOTTING_LOSS_CURVE  -----------------------------------------------


def plot_the_loss_curve(epochs, rmse):
    """Plot the loss curve, which shows loss vs. epoch."""
    plt.figure()
    plt.xlabel("Epochs")
    plt.ylabel("Root Mean Squared Error")
    plt.plot(epochs, rmse, label="Loss")
    plt.legend()
    plt.ylim([rmse.min()*0.97, rmse.max()])
    plt.show()
# -------------- PLOTTING_WEIGHT_VS_LOSS  -----------------------------------------------


def plot_the_parabola(weight, loss):
    plt.figure()
    plt.xlabel("Weight")
    plt.ylabel("Loss")
    plt.plot(weight, loss)
    plt.legend()
    # plt.ylim([rmse.min()*0.97, rmse.max()])
    plt.show()


# ===============================================================


my_feature = [1.0, 2.0,  3.0,  4.0,  5.0,  6.0,
              7.0,  8.0,  9.0, 10.0, 11.0, 12.0]
my_label = [5.0, 8.8,  9.6, 14.2, 18.8, 19.5,
            21.4, 26.8, 28.9, 32.0, 33.8, 38.2]
learning_rate = 0.3
epochs = 45
my_batch_size =18   
# ======== build the model ==========================
my_model = build_model(learning_rate)
# ========== train the model ========================
trained_weight, trained_bias, epochs, rmse, loss = train_model(
    my_model, my_feature, my_label, epochs, my_batch_size)
# ==============================================================
plot_the_model(trained_weight, trained_bias, my_feature, my_label)
plot_the_loss_curve(epochs, rmse)
# plot_the_parabola(epochs, loss)
