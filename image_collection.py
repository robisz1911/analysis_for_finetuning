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

    img_counter = 0

    for idx, [layer, size] in enumerate(layers):
        for neuron in range(int(size)):
            name = path + sub_folder_round(idx) + "/" + layer + "-Relu/" + layer + "-Relu" + "_" + str(neuron)
            print(name + "_googlenet_default.pb.png")
            img_counter += 2

            urllib.request.urlretrieve(name + "_googlenet_default.pb.png", "neuron_catalog/" + layer
                + "_" + str(neuron) + "_default.pb.png")
            urllib.request.urlretrieve(name + "_googlenet_default.pb.png", "neuron_catalog/" + layer
                + "_" + str(neuron) + "_finetuned.pb.png")
            print("Number of pics: " + str(img_counter))
    print(img_counter)
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

    x = np.load("layerObjs/Mixed_5b_Branch_3_b_1x1_act_finetuned.npy")
    print(x.shape)
    print(x[1,1,1,:])
    x = np.load("avg_col_by_neuron_for_ea_layer_default.npy", allow_pickle=True)
    print(x.shape)
    print(x)

def plot_avg_col():
    layers = get_layers()
    avgs = []

    default = np.load("avg_col_by_neuron_for_ea_layer_default.npy", allow_pickle=True)
    finetuned = np.load("avg_col_by_neuron_for_ea_layer_finetuned.npy", allow_pickle=True)

    for idx, [layer, size] in tqdm(enumerate(layers)):
        default_red  = default[idx][:,0]
        default_green = default[idx][:,1]
        default_blue = default[idx][:,2]

        finetuned_red = finetuned[idx][:,0]
        finetuned_green = finetuned[idx][:,1]
        finetuned_blue = finetuned[idx][:,2]


        plt.plot(range(size), red, label='red_d', color='r', linewidth=3)
        plt.plot(range(size), green, label='green_d', color='g', linewidth=3)
        plt.plot(range(size), blue, label='blue_d', color='b', linewidth=3)

        plt.plot(range(size), red_f, '-.', label='red_f', color='r', linewidth=3)
        plt.plot(range(size), green_f, '-.', label='green_f', color='g', linewidth=3)
        plt.plot(range(size), blue_f, '-.', label='blue_f', color='b', linewidth=3)

        plt.title(layer)
        plt.ylabel('colour_value')
        plt.xlabel('neuron')
        plt.legend()
        for i in range(10):
            plt.axvline(x=i)

        plt.xticks(range(10))
        plt.rcParams['figure.figsize'] = (4, 15)
        plt.savefig("plots/" + layer + "avgs" + '.png')



##################################################################################################


# download imgs from ai.reny.hu to local
#download_imgs()


# finetuned imgs to numpy obj (layer by layer)
#save_imgs_to_numpy(True)

# default imgs to numpy obj
#save_imgs_to_numpy(False)

avg_colour_for_each_neuron(True)
avg_colour_for_each_neuron(False)
