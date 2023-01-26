import json
from tqdm import tqdm
import urllib.request
from skimage import io
import numpy as np
import matplotlib.pyplot as plt

def sub_folder_round(number):
    if int(number + 1) < 20:
        return "1-20"
    cur = (int(number) + 1) // 10 * 10
    x = str(cur + 1) + "-" + str(min(cur + 10, 57))
    return x

# returns layers if form:
# [[layername1, size1], ... [layernameN, sizeN]]
def get_layers():
    path = "https://ai.renyi.hu/visualizing-transfer-learning/lucid/catalogs/celeba_neuron_catalog_"
    with open('layers_iter.txt', 'r') as file:
        file_concept = file.read().rstrip()
    file_concept = file_concept.replace("'", '"')
    return json.loads(file_concept)


def save_imgs_to_numpy(finetuned):
    layers = get_layers()

    for idx, [layer, size] in tqdm(enumerate(layers)):
        layer_obj = np.zeros([int(size), 256, 256, 3])
        print([layer, size])
        for neuron in range(int(size)):
            if finetuned:
                path = "neuron_catalog/" + layer + "_" + str(neuron) + "_finetuned.pb.png"
            else:
                path = "neuron_catalog/" + layer + "_" + str(neuron) + "_default.pb.png"
            # img -> numpy.ndarray
            photo = io.imread(path)
            # print([layer, size, idx])
            layer_obj[neuron, :, :, :] = photo
        # save layerObj to file
        if finetuned:
            name = "layerObjs/" + layer + "_finetuned"
        else:
            name = "layerObjs/" + layer + "_default"
        layer_obj = layer_obj.astype(int)
        np.save(name, layer_obj)
    print("Numpy objects are ready under: layersObj/")



def download_imgs():
    layers = get_layers()
    path = "https://ai.renyi.hu/visualizing-transfer-learning/lucid/catalogs/celeba_neuron_catalog_"
    img_counter = 0

    for idx, [layer, size] in tqdm(enumerate(layers)):
        for neuron in range(int(size)):
            name = path + sub_folder_round(idx) + "/" + layer + "-Relu/" + layer + "-Relu" + "_" + str(neuron)

            img_counter += 2

            urllib.request.urlretrieve(name + "_googlenet_default.pb.png", "neuron_catalog/" + layer
                + "_" + str(neuron) + "_default.pb.png")
            urllib.request.urlretrieve(name + "_googlenet_finetuned.pb.png", "neuron_catalog/" + layer
                + "_" + str(neuron) + "_finetuned.pb.png")
    print("Number of pics: " + str(img_counter))
    print("Downloading images has ended!")


# returns:
# [ [(l1_n1_avg_red, l1_n1_avg_green ,l1_n1_avg_blue), ... (l1_n1_avg_red, l1_n1_avg_green ,l1_nM1_avg_blue)],
#   .
#   .
#   .
#   [(lN_n1_avg_red, lN_n1_avg_green ,lN_n1_avg_blue), ... (lN_nMN_avg_red, lN_nMN_avg_green ,lN_nMN_avg_blue)]]
def avg_colour_for_each_neuron(finetuned):
    layers = get_layers()
    avgs = []
    for idx, [layer, size] in tqdm(enumerate(layers)):
        avg_rgb_colour_for_layer_per_neuron = np.zeros([int(size),3])

        for neuron in range(int(size)):
            if finetuned:
                postfix = "_finetuned"
                name = layer + "_finetuned.npy"
            else:
                postfix = "_default"
                name = layer + "_default.npy"

            layer_obj = np.load("layerObjs/" + name)

            red  = layer_obj[neuron,:,:,0].mean()
            green = layer_obj[neuron,:,:,1].mean()
            blue = layer_obj[neuron,:,:,2].mean()
            
            avg_rgb_colour_for_layer_per_neuron[neuron, :] =  np.array([red,green,blue])

        avgs.append(avg_rgb_colour_for_layer_per_neuron)
    np.save("avg_col_by_neuron_for_ea_layer" + postfix, avgs)

def plot_avg_col():
    layers = get_layers()
    avgs = []

    default = np.load("avg_col_by_neuron_for_ea_layer_default.npy", allow_pickle=True)
    finetuned = np.load("avg_col_by_neuron_for_ea_layer_finetuned.npy", allow_pickle=True)
    for idx, [layer, size] in tqdm(enumerate(layers)):
        plt.figure(figsize=(100, 5), dpi=1)

        default_red  = default[idx][:,0]
        default_green = default[idx][:,1]
        default_blue = default[idx][:,2]

        finetuned_red = finetuned[idx][:,0]
        finetuned_green = finetuned[idx][:,1]
        finetuned_blue = finetuned[idx][:,2]


        plt.plot(range(int(size)), default_red, label='red_d', color='r', linewidth=1)
        plt.plot(range(int(size)), default_green, label='green_d', color='g', linewidth=1)
        plt.plot(range(int(size)), default_blue, label='blue_d', color='b', linewidth=1)

        plt.plot(range(int(size)), finetuned_red, '-.', label='red_f', color='r', linewidth=2)
        plt.plot(range(int(size)), finetuned_green, '-.', label='green_f', color='g', linewidth=2)
        plt.plot(range(int(size)), finetuned_blue, '-.', label='blue_f', color='b', linewidth=2)

        plt.title(layer)
        plt.ylabel('colour_value')
        plt.xlabel('neuron')
        plt.legend()
        for i in range(int(size)):
            plt.axvline(x=i)

        plt.xticks(range(int(size)))
        plt.savefig("plots/" + layer + "avgs" + '.png', dpi = 100)
        plt.close()

def square_diff_of_avgs():
    layers = get_layers()
    
    square_diff_by_color = {"red" : [], "green" : [], "blue" : []}

    default = np.load("avg_col_by_neuron_for_ea_layer_default.npy", allow_pickle=True)
    finetuned = np.load("avg_col_by_neuron_for_ea_layer_finetuned.npy", allow_pickle=True)
    for idx, [layer, size] in tqdm(enumerate(layers)):
        default_red  = default[idx][:,0]
        default_green = default[idx][:,1]
        default_blue = default[idx][:,2]

        finetuned_red = finetuned[idx][:,0]
        finetuned_green = finetuned[idx][:,1]
        finetuned_blue = finetuned[idx][:,2]
        
        # calc square diff of avgs, and mean it over layer
#        sq_diff_red = np.absolute(default_red - finetuned_red).mean()
#        sq_diff_green = np.absolute(default_green - finetuned_green).mean()
#        sq_diff_blue = np.absolute(default_blue - finetuned_blue).mean()

        sq_diff_red = np.square(default_red - finetuned_red).mean()
        sq_diff_green = np.square(default_green - finetuned_green).mean()
        sq_diff_blue = np.square(default_blue - finetuned_blue).mean()


        square_diff_by_color["red"].append(sq_diff_red)
        square_diff_by_color["green"].append(sq_diff_green)
        square_diff_by_color["blue"].append(sq_diff_blue)

    # create plots
    plt.figure(figsize=(100, 5), dpi=1)

    plt.plot(range(len(square_diff_by_color["red"])), square_diff_by_color["red"], label='red', color='r', linewidth=1)
    plt.plot(range(len(square_diff_by_color["green"])), square_diff_by_color["green"], label='green', color='g', linewidth=1)
    plt.plot(range(len(square_diff_by_color["blue"])), square_diff_by_color["blue"], label='blue', color='b', linewidth=1)

    plt.title("Average color change by layer")
    plt.ylabel('mean of square diffs over all neurons')
    plt.xlabel('layer')
    plt.legend()
    for i in range(len(square_diff_by_color["blue"])):
        plt.axvline(x=i)
    plt.xticks(range(len(square_diff_by_color["blue"])))
    plt.savefig("stats/" + "Average_color_change_by_layer_abs_diff" + '.png', dpi = 100)
    plt.close()





##################################################################################################

# download imgs from ai.reny.hu to local
#download_imgs()

# finetuned imgs to numpy obj (layer by layer)
#save_imgs_to_numpy(True)

# default imgs to numpy obj
#save_imgs_to_numpy(False)

# finetuned
#avg_colour_for_each_neuron(True)

# default
#avg_colour_for_each_neuron(False)

# plot
#plot_avg_col()

# stat 
square_diff_of_avgs()
