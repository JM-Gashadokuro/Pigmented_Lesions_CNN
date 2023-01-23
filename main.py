import keras
import numpy as np

from PIL import Image
import PySimpleGUIQt as sgq
import os.path


model = keras.models.load_model("output/model.h5")

font = ("Arial", 20)

print("Loaded model")

lesion_type_dict = {
    0: 'Actinic keratoses',
    1: 'Basal cell carcinoma',
    2: 'Benign keratosis-like lesions ',
    3: 'Dermatofibroma',
    4: 'Melanocytic nevi',
    5: 'Melanoma',
    6: 'Vascular lesions',
}

# Left column

file_list_column = [
    [
        sgq.Text("Image Folder"),
        sgq.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sgq.FolderBrowse(),
    ],
    [
        sgq.Listbox(
            values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"
        )
    ],
]

# Right column
image_viewer_column = [
    [sgq.Text("Choose an image from list on left:")],
    [sgq.Image(key="-IMAGE-")],
    [sgq.Text(size=(50, 2), key="-TOUT-", font=font)],
]

# Layout
layout = [
    [
        sgq.Column(file_list_column),
        sgq.VSeperator(),
        sgq.Column(image_viewer_column),
    ]
]

window = sgq.Window("Image Viewer", layout)

# Run the Event Loop
while True:
    event, values = window.read()
    if event == "Exit" or event == sgq.WIN_CLOSED:
        break
    # Folder name was filled in, make a list of files in the folder
    if event == "-FOLDER-":
        folder = values["-FOLDER-"]
        try:
            # Get list of files in folder
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith(".jpg")
        ]
        window["-FILE LIST-"].update(fnames)
    elif event == "-FILE LIST-":  # A file was chosen from the listbox
        try:
            filename = os.path.join(
                values["-FOLDER-"], values["-FILE LIST-"][0]
            )
            window["-IMAGE-"].update(filename=filename)
            img_to_predict = Image.open(filename).resize((100, 75))
            img_to_predict = (np.expand_dims(img_to_predict, 0))
            prediction = model.predict(img_to_predict)
            index = np.argmax(prediction[0], 0)
            window["-TOUT-"].update("Prediction: "+lesion_type_dict[index])
            #window["-TOUT-"].update(prediction[0])

        except:
            pass

window.close()

