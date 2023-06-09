from matplotlib import pyplot as plt
import tensorflow as tf
import pandas as pd

# ========================================
# ------ Configuring_Pandas---------------------------------
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format
# ------- Loading_Dataset---------------------------------------------
# training_df = pd.read_csv(filepath_or_buffer="./california_housing_train.csv")
training_df = pd.read_csv(
    filepath_or_buffer="https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")
training_df["median_house_value"] /= 1000.0

# print(training_df.describe())
# =============================Anamoloies in the datasets =================================================

# @title Double-click to view a possible answer.

# The maximum value (max) of several columns seems very
# high compared to the other quantiles. For example,
# example the total_rooms column. Given the quantile
# values (25%, 50%, and 75%), you might expect the
# max value of total_rooms to be approximately
# 5,000 or possibly 10,000. However, the max value
# is actually 37,937.

# When you see anomalies in a column, become more careful
# about using that column as a feature. That said,
# anomalies in potential features sometimes mirror
# anomalies in the label, which could make the column
# be (or seem to be) a powerful feature.
# Also, as you will see later in the course, you
# might be able to represent (pre-process) raw data
# in order to make columns into useful features.
# =====================================================================

# ===============================================

# -------------- BUILDING_MODEL  -----------------------------------------------


def build_model(my_learning_rate):
    '''Create and compile a simple linear regression model 👌'''
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


def train_model(model, df, feature, label, epochs, batch_size):
    ''' Train the model by feeding it the data'''
    # Feed the feature values and the label values to the
    # model. The model will train for the specified number
    # of epochs, gradually learning how the feature values
    # relate to the label values.
    history = model.fit(
        x=df[feature],
        y=df[label],
        batch_size=batch_size,
        epochs=epochs
    )
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

    return trained_weight, trained_bias, epochs, rmse

# -------------- PLOTTING_MODEL  -----------------------------------------------


def plot_the_model(trained_weight, trained_bias, feature, label):
    """Plot the trained model against the training feature and label."""
    # Label the axes.
    plt.xlabel(feature)
    plt.ylabel(label)
    # Create a scatter plot from 200 random points of the dataset.
    random_examples = training_df.sample(n=200)
    # Plot the feature values vs. label values.
    plt.scatter(random_examples[feature], random_examples[label])
    # Create a red line representing the model. The red line starts
    # at coordinates (x0, y0) and ends at coordinates (x1, y1).
    x0 = 0
    y0 = trained_bias
    x1 = random_examples[feature].max()
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
# -------------- PREDICTION  -----------------------------------------------


def predict_house_values(n, feature, label):
    """Predict house values based on a feature."""
    batch = training_df[feature][10000:10000 + n]
    predicted_values = my_model.predict_on_batch(x=batch)
    print("feature   label          predicted")
    print("  value   value          value")
    print("          in thousand$   in thousand$")
    print("--------------------------------------")
    for i in range(n):
        print("%5.0f %6.0f %15.0f" % (training_df[feature][10000 + i],
                                      training_df[label][10000 + i],
                                      predicted_values[i][0]))


# ===============================================================
learning_rate = 0.03
epochs = 100
batch_size = 30
training_df["rooms_per_person"] = training_df["total_rooms"] / \
    training_df["population"]
my_feature = "median_income"
# my_feature = "population"
my_label = "median_house_value"
# ======== build the model ==========================
my_model = None
my_model = build_model(learning_rate)
# ========== train the model ========================
weight, bias, epochs, rmse = train_model(
    my_model, training_df, my_feature, my_label, epochs, batch_size)
print("\nThe learned weight for your model is %.4f" % weight)
print("The learned bias for your model is %.4f\n" % bias)
# ==============================================================
# plot_the_model(weight, bias, my_feature, my_label)
plot_the_loss_curve(epochs, rmse)
predict_house_values(20, my_feature, my_label)

# =========================================
# Define a synthetic feature
