"""import numpy as np
from keras.datasets import mnist

(train_X, train_y), (test_X, test_y) = mnist.load_data()

train_X = train_X.reshape(-1, 28*28)
test_X = test_X.reshape(-1, 28*28)

train_X = train_X / 255
test_X = test_X / 255"""

import tkinter as tk
from tkinter import filedialog

def showFrame(frame):
    frame.tkraise()

def mainMenuScreen(container):
    mainMenu = tk.Frame(container)
    mainMenu.pack_propagate(False)  # Prevent frame from resizing to fit its children

    title = tk.Label(mainMenu, text="Number Classifier", 
                  font=("Helvetica", 16, "bold"), 
                  foreground="black", 
                  width=30)
    title.pack()

    subtitle = tk.Label(mainMenu, text="Using Tensorflow, Keras & MNIST dataset",
                    font=("Helvetica", 12),
                    foreground="black")
    subtitle.pack()

    trainButton = tk.Button(mainMenu, text="Train Model", command=lambda: showFrame(container.frames["trainScreen"]))
    trainButton.pack()

    loadButton = tk.Button(mainMenu, text="Load Model", command=lambda: showFrame(container.frames["loadScreen"]))
    loadButton.pack()

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

    # Select model file (.joblib)
    modelFileLabel = tk.Label(loadScreen, text="Select model file (.joblib):")
    modelFileLabel.pack()

    modelFilePath = tk.StringVar()

    modelFileEntry = tk.Entry(loadScreen, textvariable=modelFilePath, width=30)
    modelFileEntry.pack()

    def select_file():
        file_path = filedialog.askopenfilename(filetypes=[("Joblib files", "*.joblib")])
        modelFilePath.set(file_path)

    selectFileButton = tk.Button(loadScreen, text="Browse", command=select_file)
    selectFileButton.pack()

    # Load model button
    loadModelButton = tk.Button(loadScreen, text="Load Model", command=lambda: print("Load model"))
    loadModelButton.pack()

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

    def add_layer():
        row_index = len(layer_types) + 1  # Current row index

        # Layer type dropdown
        layerType = tk.StringVar()
        layerType.set("Dense")
        layerTypeDropdown = tk.OptionMenu(layersFrame, layerType, "Dense", "Conv2D", "MaxPooling2D", "Flatten")
        layerTypeDropdown.grid(row=row_index, column=0, padx=5, pady=3, sticky="ew")
        layer_types.append(layerType)

        # Number of neurons entry (disabled for MaxPooling2D and Flatten layers)
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
        def remove_layer(index):
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
            if layer_type == "Dense" or layer_type == "Flatten":
                param_label.config(text="-")
                param_entry.config(state="disabled")
                if layer_type == "Dense":
                    neuronsEntry.config(state="normal")
                else:
                    neuronsEntry.config(state="disabled")
            elif layer_type == "Conv2D":
                param_label.config(text="Kernel size")
                param_entry.config(state="normal")
            elif layer_type == "MaxPooling2D":
                param_label.config(text="Pool size")
                param_entry.config(state="normal")
                neuronsEntry.config(state="disabled")
            
            # Disable activation dropdown for layers that do not use it
            if layer_type in ["MaxPooling2D", "Flatten"]:
                activationDropdown.config(state="disabled")
            else:
                activationDropdown.config(state="normal")
            
        layerType.trace_add("write", lambda *args: update_layer_fields())

        # Initialize the parameter field based on the default layer type ("Dense")
        update_layer_fields()

    add_layer()  # Add initial layer

    addLayerButton = tk.Button(trainScreen, text="Add Layer", command=add_layer)
    addLayerButton.pack(pady=5)

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
        
        container.frames["trainingScreen"] = trainingScreen(container, final_layers)
        container.frames["trainingScreen"].grid(row=0, column=0, sticky="nsew")
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


def trainingScreen(container, layers):
    trainingScreen = tk.Frame(container, bg="#f0f0f0")
    trainingScreen.pack_propagate(False)

    title = tk.Label(
        trainingScreen, 
        text="Training Model", 
        font=("Helvetica", 16, "bold"), 
        foreground="#333", 
        bg="#f0f0f0", 
        width=30, 
    )
    title.pack(pady=(10, 20))

    # Display the layers
    for layer in layers:
        layer_info = f"Layer {layer['Layer']}: {layer['Type']} - Neurons: {layer['Neurons']} - Parameter: {layer['Parameter']} - Activation: {layer['Activation']}"
        layer_label = tk.Label(trainingScreen, text=layer_info, bg="#f0f0f0", font=("Helvetica", 10))
        layer_label.pack(pady=5)

    # Back Button
    backButton = tk.Button(
        trainingScreen, 
        text="Back", 
        command=lambda: showFrame(container.frames["trainScreen"]),
        bg="#ddd", 
        activebackground="#bbb", 
        font=("Helvetica", 10)
    )
    backButton.pack(side="bottom", pady=(20, 10))

    return trainingScreen

def main():
    global root
    root = tk.Tk()
    root.title("Number Classifier")
    root.geometry("800x400")

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