import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

import h5py
import keras
from keras.datasets import mnist
from threading import Thread

import numpy as np

def showFrame(frame):
    frame.tkraise()

def mainMenuScreen(container):
    mainMenu = tk.Frame(container)
    mainMenu.pack_propagate(False)  # Prevent frame from resizing to fit its children

    title = tk.Label(mainMenu, text="Number Classifier", 
                  font=("Helvetica", 16, "bold"), 
                  foreground="black", 
                  width=30)
    title.pack(pady=20)

    subtitle = tk.Label(mainMenu, text="Using Keras & MNIST dataset",
                    font=("Helvetica", 12),
                    foreground="black")
    subtitle.pack(pady=5)

    trainButton = tk.Button(mainMenu, text="Train Model", command=lambda: showFrame(container.frames["trainScreen"]))
    trainButton.pack(pady=10)

    loadButton = tk.Button(mainMenu, text="Load Model", command=lambda: showFrame(container.frames["loadScreen"]))
    loadButton.pack(pady=10)

    # Image of a random number from the MNIST dataset
    imageFrame = tk.Frame(mainMenu)
    imageFrame.pack(pady=20)

    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    import random
    random_image = train_X[random.randint(0, len(train_X))]

    canvas = tk.Canvas(imageFrame, width=28*5, height=28*5)
    canvas.pack()

    for i in range(28):
        for j in range(28):
            color = "#%02x%02x%02x" % (random_image[i][j], random_image[i][j], random_image[i][j])
            canvas.create_rectangle(j*5, i*5, j*5+5, i*5+5, fill=color, outline="")

    # Exit button
    exitButton = tk.Button(mainMenu, text="Exit", command=root.quit)
    exitButton.pack(pady=10)

    return mainMenu

def loadScreen(container):
    loadScreen = tk.Frame(container)
    loadScreen.pack_propagate(False)

    title = tk.Label(loadScreen, text="Load Model", 
                  font=("Helvetica", 16, "bold"), 
                  foreground="black", 
                  width=30, 
                  height=2)
    title.pack()

    # Select model file (.h5)
    modelFileLabel = tk.Label(loadScreen, text="Select model file (.h5):", font=("Helvetica", 12))
    modelFileLabel.pack(pady=10)

    modelFilePath = tk.StringVar()

    modelFileEntry = tk.Entry(loadScreen, textvariable=modelFilePath, width=30)
    modelFileEntry.pack(pady=10)

    def select_file():
        file_path = filedialog.askopenfilename(filetypes=[("HDF5 files", "*.h5")])
        modelFilePath.set(file_path)

    selectFileButton = tk.Button(loadScreen, text="Browse", command=select_file)
    selectFileButton.pack(pady=10)

    # Load model
    def load_model():
        file_path = modelFilePath.get()
        if file_path:
            model = keras.models.load_model(file_path)
            # Read metadata
            with h5py.File(file_path, "r") as f:
                accuracy = f.attrs["accuracy"]
            container.frames["predictionScreen"] = predictionScreen(container, model, accuracy)
            container.frames["predictionScreen"].grid(row=0, column=0, sticky="nsew")

            # Clear the model file path
            modelFilePath.set("")
            showFrame(container.frames["predictionScreen"])

    loadModelButton = tk.Button(loadScreen, text="Load Model", command=load_model)
    loadModelButton.pack(pady=10)

    # Function to update the state of the load model button
    def update_button_state(*args):
        if modelFilePath.get():
            loadModelButton.config(state=tk.NORMAL)
        else:
            loadModelButton.config(state=tk.DISABLED)

    # Trace changes to the modelFilePath variable
    modelFilePath.trace_add("write", update_button_state)

    # Initially disable the load model button
    loadModelButton.config(state=tk.DISABLED)

    backButton = tk.Button(
        loadScreen, 
        text="Back", 
        command=lambda: showFrame(container.frames["mainMenu"]),
        bg="#ddd", 
        activebackground="#bbb", 
        font=("Helvetica", 10)
    )
    backButton.pack(side="bottom", pady=(20, 10))

    return loadScreen

def trainScreen(container):

    trainScreen = tk.Frame(container, bg="#f0f0f0")  # Light background color
    trainScreen.pack_propagate(False)

    title = tk.Label(
        trainScreen, 
        text="Train new model", 
        font=("Helvetica", 16, "bold"), 
        foreground="#333", 
        bg="#f0f0f0", 
        width=30, 
    )
    title.pack()  # Add padding for space around the title

    # Frame to contain the layer rows
    layersFrame = tk.Frame(trainScreen, bg="#f0f0f0")
    layersFrame.pack()

    headers = ["Layer type", "Neuron #", "Parameter","", "Activation func.", ""]
    for col, text in enumerate(headers):
        tk.Label(
            layersFrame, 
            text=text, 
            bg="#f0f0f0", 
            font=("Helvetica", 10, "bold"), 
            anchor="center"
        ).grid(row=0, column=col, padx=5, pady=3, sticky="ew")

    # Lists to store layer data
    layer_types = []
    neuron_counts = []
    parameters = []
    activation_functions = []
    remove_buttons = []

    def remove_layer(index, end=False):
        if len(layer_types) > 1 or end:
            # Remove the layer at the given index
            layer_types.pop(index-1)
            neuron_counts.pop(index-1)
            parameters.pop(index-1)
            activation_functions.pop(index-1)
            remove_buttons.pop(index-1)

            # Destroy all widgets in the row
            for widget in layersFrame.grid_slaves():
                if int(widget.grid_info()["row"]) == index:
                    widget.destroy()
                elif int(widget.grid_info()["row"]) > index:
                    widget.grid(row=widget.grid_info()["row"]-1)
                    # Update the remove button command to pass the new index
                    if widget in remove_buttons:
                        widget.config(command=lambda idx=widget.grid_info()["row"]: remove_layer(idx))

    def add_layer():

        if len(layer_types) >= 7:
            tk.messagebox.showerror("Error", "Maximum number of layers reached")
            return

        row_index = len(layer_types) + 1  # Current row index

        # Layer type dropdown
        layerType = tk.StringVar()
        layerType.set("Dense")
        layerTypeDropdown = tk.OptionMenu(layersFrame, layerType, "Dense", "Conv2D", "MaxPooling2D")
        layerTypeDropdown.grid(row=row_index, column=0, padx=5, pady=3, sticky="ew")
        layer_types.append(layerType)

        # Number of neurons entry (disabled for MaxPooling2D layers)
        neurons = tk.StringVar()
        neuronsEntry = tk.Entry(layersFrame, textvariable=neurons, width=5)
        neuronsEntry.grid(row=row_index, column=1, padx=5, pady=3, sticky="ew")
        neuron_counts.append(neurons)

        # Parameter label and entry
        param_label = tk.Label(layersFrame, text="-", bg="#f0f0f0")
        param_label.grid(row=row_index, column=2, padx=5, pady=3, sticky="ew")
        param_entry = tk.Entry(layersFrame, state="disabled", width=5)
        param_entry.grid(row=row_index, column=3, padx=5, pady=3, sticky="ew")
        parameters.append((param_label, param_entry))

        # Activation function dropdown (to be removed for certain layers)
        activation = tk.StringVar()
        activation.set("None")
        activationDropdown = tk.OptionMenu(layersFrame, activation, "None", "relu", "sigmoid", "softmax", "tanh")
        activationDropdown.grid(row=row_index, column=4, padx=5, pady=3, sticky="ew")
        activation_functions.append(activation)

        # Remove button (disabled for the first layer)
        


        removeLayerButton = tk.Button(layersFrame, text="X", command=lambda idx=row_index: remove_layer(idx), bg="#f00", fg="#fff", width=5)
        removeLayerButton.grid(row=row_index, column=5, padx=5, pady=3)
        remove_buttons.append(removeLayerButton)

        # Update parameter field, activation dropdown, and reset values when layer type changes
        def update_layer_fields():
            layer_type = layerType.get()
            
            # Reset neuron count and parameter field
            neurons.set("")
            param_entry.config(state="normal")
            param_entry.delete(0, tk.END)
            activation.set("None")
            if layer_type == "Dense":
                param_label.config(text="-")
                param_entry.config(state="disabled")
                neuronsEntry.config(state="normal")
            elif layer_type == "Conv2D":
                param_label.config(text="Kernel size")
                param_entry.config(state="normal")
            elif layer_type == "MaxPooling2D":
                param_label.config(text="Pool size")
                param_entry.config(state="normal")
                neuronsEntry.config(state="disabled")
            
            # Disable activation dropdown for layers that do not use it
            if layer_type  == "MaxPooling2D":
                activationDropdown.config(state="disabled")
            else:
                activationDropdown.config(state="normal")
            
        layerType.trace_add("write", lambda *args: update_layer_fields())

        # Initialize the parameter field based on the default layer type ("Dense")
        update_layer_fields()

    add_layer()  # Add initial layer

    addLayerButton = tk.Button(trainScreen, text="Add Layer", command=add_layer)
    addLayerButton.pack(pady=5)

    # Epochs entry
    epochsLabel = tk.Label(trainScreen, text="Epochs:")
    epochsLabel.pack()

    epochs = tk.StringVar()
    epochsEntry = tk.Entry(trainScreen, textvariable=epochs, width=5)
    epochsEntry.pack()

    # Train button function
    def train_model():
        final_layers = []
        for i in range(len(layer_types)):
            layer_info = {
                "Layer": i + 1,
                "Type": layer_types[i].get(),
                "Neurons": neuron_counts[i].get(),
                "Parameter": parameters[i][1].get() if parameters[i][1].get() != "-" else "N/A",
                "Activation": activation_functions[i].get() if layer_types[i].get() in ["Dense", "Conv2D"] else "N/A"
            }
            # Check if the layer is valid
            valid = True
            if layer_info["Type"] == "Dense":
                if not layer_info["Neurons"] or layer_info["Activation"]=="None":
                    valid = False
            elif layer_info["Type"] == "Conv2D":
                if not layer_info["Neurons"] or not layer_info["Parameter"] or layer_info["Activation"]=="None":
                    valid = False
            elif layer_info["Type"] == "MaxPooling2D":
                if not layer_info["Parameter"]:
                    valid = False
            
            if not valid:
                # Display an error message
                tk.messagebox.showerror("Error", f"Fill all required fields")
                return
            final_layers.append(layer_info)

        # Check if the epochs field is filled
        if not epochs.get():
            tk.messagebox.showerror("Error", "Fill the epochs field")
            return
        

        container.frames["trainingScreen"] = trainingScreen(container, final_layers, int(epochs.get()))
        container.frames["trainingScreen"].grid(row=0, column=0, sticky="nsew")

        # Remove layers
        for i in range(len(layer_types), 0, -1):
            remove_layer(i, end=True)
        
        add_layer()  # Add initial layer

        # Reset epochs field
        epochs.set("")


        showFrame(container.frames["trainingScreen"])


    trainButton = tk.Button(trainScreen, text="Train", command=train_model, bg="#4CAF50", fg="white", font=("Helvetica", 10, "bold"))
    trainButton.pack()

    # Back Button
    backButton = tk.Button(
        trainScreen, 
        text="Back", 
        command=lambda: showFrame(container.frames["mainMenu"]),
        bg="#ddd", 
        activebackground="#bbb", 
        font=("Helvetica", 10)
    )
    backButton.pack(side="bottom", pady=(20, 10))

    return trainScreen

def trainingScreen(container, layers, epochs):
    trainingScreen = tk.Frame(container, bg="#f0f0f0")
    trainingScreen.pack_propagate(False)

    title = tk.Label(
        trainingScreen, 
        text="Training Model...", 
        font=("Helvetica", 16, "bold"), 
        foreground="#333", 
        bg="#f0f0f0", 
        width=30, 
    )
    title.pack(pady=(10, 20))

    progress_bar = ttk.Progressbar(trainingScreen, orient="horizontal", length=300, mode="determinate")
    progress_bar.pack(pady=(10, 20))
    progress_bar["maximum"] = epochs

    predict_button = None

    def train_model_thread():
        model, accuracy = trainModel(layers, epochs, progress_bar)

        predict_button = tk.Button(
            trainingScreen, 
            text="Predict", 
            font=("Helvetica", 14), 
            command=lambda: go_to_predictionScreen(model, accuracy)
        )
        predict_button.pack(pady=(20, 0))

        progress_bar["value"] = epochs
    
    def go_to_predictionScreen(model, accuracy):
        # Transition to the next screen and pass the model and accuracy
        container.frames["predictionScreen"] = predictionScreen(container, model, accuracy)
        container.frames["predictionScreen"].grid(row=0, column=0, sticky="nsew")
        showFrame(container.frames["predictionScreen"])

    thread = Thread(target=train_model_thread)
    thread.start()

    return trainingScreen

def trainModel(layers, epochs, progress_bar):
    layerList = []

    for layer in layers:
        if layer["Type"] == "Dense":
            layerList.append(keras.layers.Dense(int(layer["Neurons"]), activation=layer["Activation"]))
        elif layer["Type"] == "Conv2D":
            layerList.append(keras.layers.Conv2D(int(layer["Neurons"]), kernel_size=int(layer["Parameter"]), activation=layer["Activation"]))
        elif layer["Type"] == "MaxPooling2D":
            layerList.append(keras.layers.MaxPooling2D(pool_size=int(layer["Parameter"])))

    model = keras.Sequential(layerList)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    train_X = train_X.reshape(-1, 28*28)
    test_X = test_X.reshape(-1, 28*28)

    train_X = train_X / 255
    test_X = test_X / 255

    for epoch in range(epochs):
        model.fit(train_X, train_y, epochs=1, verbose=0)
        progress_bar["value"] = epoch + 1
        root.update_idletasks()

    test_loss, test_acc = model.evaluate(test_X, test_y)

    return model, test_acc

def predictionScreen(container, model, accuracy):

    predictionScreen = tk.Frame(container, bg="#f0f0f0")
    predictionScreen.pack_propagate(False)

    title = tk.Label(
        predictionScreen, 
        text="Predict", 
        font=("Helvetica", 16, "bold"), 
        foreground="#333", 
        bg="#f0f0f0", 
        width=30, 
    )
    title.pack()

    accuracy_label = tk.Label(
        predictionScreen, 
        text=f"Accuracy: {accuracy:.4f}", 
        font=("Helvetica", 14), 
        fg="#333", 
        bg="#f0f0f0"
    )
    accuracy_label.pack(pady=5)

    # Frame to contain the probability bars and canvas
    main_frame = tk.Frame(predictionScreen, bg="#f0f0f0")
    main_frame.pack(pady=5, side="top")

    canvas_size = 8

    # Canvas for drawing the number
    canvas = tk.Canvas(main_frame, width=28*canvas_size, height=28*canvas_size, bg="black", bd=2, relief="solid")
    canvas.pack(side="left", padx=20)

    points = [[0]*28 for _ in range(28)]

    def draw(event):
        # Convert the canvas coordinates to grid coordinates
        x = event.x // canvas_size
        y = event.y // canvas_size

        if 0 <= x < 28 and 0 <= y < 28:
            
            points[y][x] = 255  # Set pixel to white for drawing
            canvas.create_rectangle(x*canvas_size, y*canvas_size, (x+1)*canvas_size, (y+1)*canvas_size, fill="white", outline="")

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Draw on the neighboring pixels for continuity
                nx, ny = x + dx, y + dy
                if 0 <= nx < 28 and 0 <= ny < 28 and points[ny][nx] < 170:
                    points[ny][nx] = 170
                    canvas.create_rectangle(nx*canvas_size, ny*canvas_size, (nx+1)*canvas_size, (ny+1)*canvas_size, fill="lightgray", outline="")

            for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < 28 and 0 <= ny < 28 and points[ny][nx] < 85:
                    points[ny][nx] = 85
                    canvas.create_rectangle(nx*canvas_size, ny*canvas_size, (nx+1)*canvas_size, (ny+1)*canvas_size, fill="gray", outline="")

    canvas.bind("<B1-Motion>", draw)

    def clear_canvas():
        canvas.delete("all")
        for i in range(28):
            for j in range(28):
                points[i][j] = 0
        reset_prediction()  # Reset prediction when clearing the canvas

    # Frame to hold prediction probabilities
    probability_frame = tk.Frame(main_frame, bg="#f0f0f0")
    probability_frame.pack(side="right", padx=20)

    # Display prediction probabilities and highlight the highest probability
    probability_labels = []
    bars = []
    for i in range(10):
        label = tk.Label(probability_frame, text=f"{i} -", font=("Helvetica", 12), fg="#333", bg="#f0f0f0")
        label.grid(row=i, column=0, pady=1, sticky="w")
        probability_labels.append(label)

        bar = tk.Canvas(probability_frame, height=10, width=200, bg="lightgray", bd=0, highlightthickness=0)
        bar.grid(row=i, column=1, pady=1, sticky="w")
        bars.append(bar)

    # To reset the display of probabilities
    def reset_prediction():
        for bar in bars:
            bar.delete("all")
            bar.create_rectangle(0, 0, 0, 10, fill="gray")

    # Real-time prediction function
    def predict_digit():

        image_data = np.array(points)

        image_data = image_data.reshape(1, 28*28).astype("float32")
        image_data = image_data / 255.0

        # Predict the digit
        prediction = model.predict(image_data)        
        
        # Display probabilities and highlight the highest probability
        max_prob_digit = np.argmax(prediction)
        for i, prob in enumerate(prediction[0]):
            prob_percentage = prob * 100
            
            # Update probability bar
            bars[i].delete("all")
            if i == max_prob_digit:
                bars[i].create_rectangle(0, 0, prob_percentage * 2, 10, fill="blue")
            else:
                bars[i].create_rectangle(0, 0, prob_percentage * 2, 10, fill="lightblue")

    # Function for the prediction button click
    def on_predict_button_click():
        predict_digit()

    # Create a frame to hold the buttons
    button_frame = tk.Frame(predictionScreen)
    button_frame.pack(pady=(5, 5))

    # Buttons
    clear_button = tk.Button(button_frame, text="Clear", font=("Helvetica", 12), command=clear_canvas)
    clear_button.pack(side=tk.LEFT, padx=(5, 10))

    predict_button = tk.Button(button_frame, text="Predict", font=("Helvetica", 12), command=on_predict_button_click)
    predict_button.pack(side=tk.LEFT, padx=(10, 5))

    # Save model button
    def save_model():
        # As .h5 file
        file_path = filedialog.asksaveasfilename(filetypes=[("HDF5 files", "*.h5")])
        if file_path:
            model.save(file_path+".h5")
            with h5py.File(file_path+".h5", "a") as f:
                f.attrs["accuracy"] = accuracy
            

    save_button = tk.Button(predictionScreen, text="Save Model", font=("Helvetica", 12), command=save_model)
    save_button.pack(pady=(5, 10))

    backButton = tk.Button(
        predictionScreen, 
        text="Main Menu", 
        command=lambda: showFrame(container.frames["mainMenu"]),
        bg="#ddd", 
        activebackground="#bbb", 
        font=("Helvetica", 12)
    )
    backButton.pack(side="bottom", pady=(20, 10))

    return predictionScreen

def main():
    global root
    root = tk.Tk()
    root.title("Number Classifier")
    root.geometry("800x500")

    # Create a container for the frames
    container = tk.Frame(root)
    container.pack(side="top", fill="both", expand=True)

    # Configure the grid layout
    container.grid_rowconfigure(0, weight=1)
    container.grid_columnconfigure(0, weight=1)

    # Create frames dictionary
    container.frames = {}

    # Create and store frames
    container.frames["mainMenu"] = mainMenuScreen(container)
    container.frames["loadScreen"] = loadScreen(container)
    container.frames["trainScreen"] = trainScreen(container)

    for frame in container.frames.values():
        frame.grid(row=0, column=0, sticky="nsew")

    # Show the first frame
    showFrame(container.frames["mainMenu"])

    root.mainloop()

if __name__ == "__main__":
    main()