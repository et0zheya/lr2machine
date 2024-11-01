import os

from dotenv import load_dotenv

from . import read_one_img_to_matrix, neuron_net


def recognition(img_path):
    img_matrix = read_one_img_to_matrix(img_path)

    load_dotenv()
    scales_index = int(os.getenv('scales_index'))
    layer_matrices = img_matrix.iloc[0]

    recognizer = neuron_net(layer_matrices, scales_index)[0]
    get_answer = recognizer.idxmax()
    services = {
        0: "Ae",
        1: "Ai",
        2: "Editor",
        3: "Fi",
        4: "Figma",
        5: "Id",
        6: "Paint",
        7: "Pr",
        8: "V",
        9: "Wing"
    }

    return services.get(get_answer, "Invalid value. Please enter a number from 0 to 9.")
